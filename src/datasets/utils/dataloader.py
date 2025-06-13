# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import csv
import io
import time

import numpy as np
import torch
from torch.utils.data import _utils
from torch.utils.data.dataloader import ExceptionWrapper, _DatasetKind, _MultiProcessingDataLoaderIter

from src.utils.monitoring import ResourceMonitoringThread


class ConcatIndices:
    """Helper to map indices of concatenated/mixed datasets to the sample index for the corresponding dataset."""

    cumulative_sizes: np.ndarray

    def __init__(self, sizes):
        self.cumulative_sizes = np.cumsum(sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        # Returns a pair (dataset_idx, sample_idx)
        if idx < 0 or idx >= len(self):
            raise ValueError(f"index must be between 0 and the total size ({len(self)})")
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            return dataset_idx, idx
        return dataset_idx, idx - self.cumulative_sizes[dataset_idx - 1]


class CSVLogger(object):
    """An append-to CSV abstraction. File I/O requires a flush."""

    def __init__(self, fname, header):
        """Write header to internal buffers."""
        self.fname = fname
        self.buffer = io.StringIO()
        self.writer = csv.writer(self.buffer, quoting=csv.QUOTE_NONNUMERIC)
        self.writer.writerow(header)
        self.initialized = False

    def writerow(self, row) -> None:
        """Write row to internal buffers."""
        self.writer.writerow(row)

    def flush(self) -> None:
        """Flush buffer to file."""
        # Overwrite old file
        mode = "a+" if self.initialized else "w"

        with open(self.fname, mode, newline="") as f:
            f.write(self.buffer.getvalue())

        self.buffer = io.StringIO()
        self.writer = csv.writer(self.buffer, quoting=csv.QUOTE_NONNUMERIC)
        self.initialized = True


class MonitoredDataset(torch.utils.data.Dataset):
    """Implement resource monitoring on a per-worker basis.

    The sampling occurs every monitor_interval seconds and writes the log
    every log_interval seconds to a file specified by log_filename, which
    maps a worker id to a file using the '%w' placeholder.

    Warning: Do not call this dataset before it is consumed in the DataLoader.
    """

    def __init__(
        self, dataset: torch.utils.data.Dataset, log_filename: str, log_interval: float, monitor_interval: float
    ):
        self.dataset = dataset
        self.log_filename = str(log_filename)
        self.log_interval = log_interval
        self.monitor_interval = monitor_interval
        self._csv_log = None
        self._monitoring_thread = None
        self._last_log_time = None
        # Patch getitems dynamically
        if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:

            def __getitems__(self, index):
                self.maybe_start_resource_monitoring()
                return self.dataset.__getitems__(index)

            self.__getitems__ = __getitems__

    def __del__(self):
        self.stop_resource_monitoring()

    def __getitem__(self, index):
        self.maybe_start_resource_monitoring()
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def _elapsed_log_time(self):
        if self._last_log_time is None:
            return float("inf")
        else:
            return time.perf_counter() - self._last_log_time

    def _update_log_time(self):
        self._last_log_time = time.perf_counter()

    def maybe_start_resource_monitoring(self):
        if self._monitoring_thread is None:

            def callback_fn(resource_sample):
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id

                if self._csv_log is None:
                    header = [f.name for f in resource_sample.fields()]
                    log_filename = self.log_filename.replace("%w", str(worker_id))
                    self._csv_log = CSVLogger(log_filename, header)
                row_values = resource_sample.as_tuple()
                self._csv_log.writerow(row_values)

                if self._elapsed_log_time() > self.log_interval:
                    self._csv_log.flush()
                    self._update_log_time()

            self._monitoring_thread = ResourceMonitoringThread(
                None, self.monitor_interval, stats_callback_fn=callback_fn
            )
            self._monitoring_thread.start()

    def stop_resource_monitoring(self):
        if self._monitoring_thread:
            self._monitoring_thread.stop()


class NondeterministicDataLoader(torch.utils.data.DataLoader):
    """Override torch dataloader to return out of order."""

    def __init__(self, *args, **kwargs):
        """Pass through constructor."""
        super().__init__(*args, **kwargs)

    def _get_iterator(self):
        if self.num_workers:
            self.check_worker_number_rationality()
            return _SloppyMultiProcessingDataLoaderIter(self)
        else:
            return super()._get_iterator()


class _SloppyMultiProcessingDataLoaderIter(_MultiProcessingDataLoaderIter):

    def __init__(self, *args, **kwargs):
        """Pass through constructor."""
        super().__init__(*args, **kwargs)

    def _next_data(self):
        """Adds out of order returns."""
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                if info is None:
                    # Found a reordered tombstone
                    del self._task_info[self._rcvd_idx]
                    self._rcvd_idx += 1
                    self._try_put_index()
                else:
                    worker_id = info[0]
                    # has data or is still active
                    if len(info) == 2 or self._workers_status[worker_id]:
                        break
                    del self._task_info[self._rcvd_idx]
                    self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # Tombstone to receive later
                self._task_info[idx] = None
                if isinstance(data, ExceptionWrapper):
                    data.reraise()
                return data
            else:
                del self._task_info[idx]
                return self._process_data(data)


def get_worker_info():
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        num_workers = 1
        worker_id = 0
    else:
        num_workers = worker_info.num_workers
        worker_id = worker_info.id
    return num_workers, worker_id
