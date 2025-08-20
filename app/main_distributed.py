# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import datetime
import os
import pprint
import shutil
from pathlib import Path

import submitit
import yaml

from app.scaffold import main as app_main
from src.utils.logging import get_logger, git_information

logger = get_logger(force=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    help="yaml file containing config file names to launch",
    default="configs.yaml",
)
parser.add_argument("--exclude", type=str, help="nodes to exclude from training", default=None)
parser.add_argument(
    "--batch-launch",
    action="store_true",
    help="whether fname points to a file to batch-launch several config files",
)
parser.add_argument(
    "--use_fname_as_folder",
    action="store_true",
    help="whether to append fname filename to folder",
)
parser.add_argument(
    "--folder",
    type=str,
    default=None,
    help="if specified, override 'folder' field in the .yaml with this",
)
parser.add_argument(
    "--account",
    type=str,
    default="jepa",
    help="Cluster account to use when submitting jobs",
)
parser.add_argument(
    "--partition",
    type=str,
    default="learn",
    help="Cluster partition to use when submitting jobs",
)
parser.add_argument(
    "--qos",
    type=str,
    default=None,
    help="If specified, cluster partition to use when submitting jobs",
)
parser.add_argument("--time", type=int, default=4300, help="time in minutes to run job")


class Trainer:
    def __init__(self, args_pretrain, load_model=None):
        self.app = args_pretrain["app"]
        self.args_pretrain = args_pretrain
        self.load_model = load_model

    def __call__(self):
        app = self.app
        params = self.args_pretrain
        load_model = self.load_model

        logger.info("loaded pretrain params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

        # Launch app with loaded config
        resume_preempt = False if load_model is None else load_model
        app_main(app, args=params, resume_preempt=resume_preempt)

    def checkpoint(self):
        fb_trainer = Trainer(self.args_pretrain, True)
        return submitit.helpers.DelayedSubmission(
            fb_trainer,
        )


def copy_code_folder(code_folder, ignore_patterns, ignore_paths):
    path_to_node_folder = {}

    for path in ignore_paths:
        split_path = path.split("/")
        base_path = "/".join(split_path[:-1])
        node_folder = split_path[-1]
        path_to_node_folder[base_path] = node_folder

    def ignore_func(path, names):
        ignore_list = ignore_patterns
        if path in path_to_node_folder.keys():
            ignore_list.append(path_to_node_folder[path])
        return ignore_list

    if not os.path.exists(code_folder):
        shutil.copytree(".", code_folder, ignore=ignore_func)


def update_folder_with_timestamp(args_list):
    new_args_list = copy.deepcopy(args_list)
    for i, args in enumerate(args_list):
        folder = args["folder"]
        load_checkpoint = args["meta"].get("load_checkpoint", False) if "meta" in args else False
        if not load_checkpoint and Path(folder).exists():
            timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
            folder = folder.rstrip("/") + f"_{timestamp}"
            logger.info(f"Folder already exists but `load_checkpoint` is False. Logging to new folder {folder}...")
            new_args_list[i]["folder"] = folder
    return new_args_list


def launch_app_with_parsed_args(
    args_for_pretrain,
    account,
    partition,
    qos,
    mem_per_gpu="210G",
    timeout=4300,
    nodes=1,
    tasks_per_node=1,
    cpus_per_task=12,
    exclude_nodes=None,
):
    args_for_pretrain = update_folder_with_timestamp(args_for_pretrain)
    for ap in args_for_pretrain:
        folder = ap["folder"]
        Path(folder).mkdir(parents=True, exist_ok=True)
    folder = args_for_pretrain[0]["folder"]

    # -------------- Copy code --------------
    code_folder = os.path.join(folder, "code")
    ignore_patterns = [
        "__pycache__",
        ".vscode",
        ".git",
        "core",
    ]
    ignore_paths = [
        "./evals/ava/alphaction/data",
        "./demos",
        "./traces",
    ]
    copy_code_folder(code_folder, ignore_patterns, ignore_paths)
    os.chdir(code_folder)
    # ---------------------------------------

    # -------------- Save config file --------------
    params_path = os.path.join(folder, "params-pretrain.yaml")
    if not os.path.exists(params_path):
        with open(params_path, "w") as f:
            yaml.dump(args_for_pretrain, f)
    # ----------------------------------------------

    # -------------- Save git info file --------------
    git_info_fpath = os.path.join(folder, "git-info.txt")
    with open(git_info_fpath, "w") as f:
        f.write(git_information())
    # ----------------------------------------------

    # -------------- SET JOB NAME --------------
    folder_ = folder
    if folder[-1] == "/":
        folder_ = folder[:-1]
    job_name = folder_.split("/")[-1]
    # ------------------------------------------

    executor = submitit.AutoExecutor(folder=os.path.join(folder, "job_%j"), slurm_max_num_timeout=20)
    executor.update_parameters(
        name=job_name,
        slurm_partition=partition,
        slurm_account=account,
        slurm_qos=qos,
        slurm_mem_per_gpu=mem_per_gpu,
        timeout_min=timeout,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        gpus_per_node=tasks_per_node,
    )

    if exclude_nodes is not None:
        executor.update_parameters(slurm_exclude=exclude_nodes)

    jobs, trainers = [], []
    with executor.batch():
        for ap in args_for_pretrain:
            # TODO Create sub folder and ap['folder']=subfolder
            fb_trainer = Trainer(ap)
            job = executor.submit(
                fb_trainer,
            )
            trainers.append(fb_trainer)
            jobs.append(job)

    for job in jobs:
        print(job.job_id)


def launch():
    # ---------------------------------------------------------------------- #
    # 1. Put config file names in a list
    # ---------------------------------------------------------------------- #
    config_fnames = [args.fname]

    # -- If batch-launch is True, then the args.fname yaml file is not a
    # -- config, but actually specifies a list of other config files
    # -- to run in a slurm job array
    if args.batch_launch:
        with open(args.fname, "r") as y_file:
            config_fnames = yaml.load(y_file, Loader=yaml.FullLoader)
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 2. Parse each yaml config file as a dict and place in list
    # ---------------------------------------------------------------------- #
    nodes, tasks_per_node = None, None
    configs = []
    for f in config_fnames:
        with open(f, "r") as y_file:
            _params = yaml.load(y_file, Loader=yaml.FullLoader)
            if args.use_fname_as_folder:
                assert not args.folder, "Don't specify --folder if adding fname to folder"
                _params["folder"] = str(Path(_params["folder"]) / f.split("/")[-1].split(".yaml")[0])
            elif args.folder:
                _params["folder"] = args.folder
            nodes = int(_params.get("nodes"))
            tasks_per_node = int(_params.get("tasks_per_node"))
            cpus_per_task = int(_params.get("cpus_per_task", 32))
            mem_per_gpu = _params.get("mem_per_gpu", "210G")
            configs += [_params]
    logger.info(f"Loaded {len(configs)} config files")
    logger.info(f"Running all jobs with {nodes=} / {tasks_per_node=}")
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # 3. Launch evals with parsed config files
    # ---------------------------------------------------------------------- #
    launch_app_with_parsed_args(
        args_for_pretrain=configs,
        account=args.account,
        partition=args.partition,
        qos=args.qos,
        mem_per_gpu=mem_per_gpu,
        cpus_per_task=cpus_per_task,
        timeout=args.time,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        exclude_nodes=args.exclude,
    )
    # ---------------------------------------------------------------------- #


if __name__ == "__main__":
    args = parser.parse_args()
    launch()
