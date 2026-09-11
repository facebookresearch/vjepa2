# DROID physical-time sampling

Raw DROID MP4 playback FPS can differ from observation acquisition frequency. A
60 FPS MP4 does not imply that 15 frame indices represent 0.25 physical seconds.
The example training config uses `temporal_sampling: physical`. Existing configs
that omit this option retain `legacy` sampling for experiment compatibility;
that mode uses MP4 playback FPS and must not be described as physical FPS.

Physical mode uses the native `<camera>_timestamps.json` beside each MP4 and the
matching `observation/timestamp/cameras/<camera>_frame_received` HDF5 dataset.
Both contain integer milliseconds in the ZED IMAGE clock. Every video timestamp
must map exactly to a unique, ordered robot observation row. Missing video frames
therefore do not shift action/state indices. Sampled pose differences retain the
existing translation, Euler rotation-difference, and gripper-difference convention.

If sidecars are missing, obtain/export the original per-frame IMAGE timestamps
from the SVO recordings. **Do not generate a purported native sidecar from MP4 FPS
or copy HDF5 row timestamps without proving frame identity.** Physical mode fails
clearly on missing or inconsistent clock metadata; it does not silently fall back.
For an independently established frame-i -> observation-i export, an explicit
`timestamp_alignment: row` compatibility option permits N or N+1 HDF5 rows for N
video frames. This is a caller assertion, not a proof derived from cardinality.
An existing native sidecar always takes precedence, including in row mode.

Each clip is anchored at a real frame, with target times `t0 + k / fps`. The
nearest distinct observed frames are selected, preferring the earlier frame on
a tie. Targets require full coverage, maximum absolute error
`timestamp_tolerance_s` (default 0.05 seconds), and no source-observation gap above
`max_observation_gap_s` (default 0.25 seconds), including gaps between targets.
These are explicit data-quality policies and may be configured for another
acquisition protocol. No images, poses, or actions are interpolated. Eight images
at 4 physical FPS have a 1.75-second first-to-last target span.

Physical mode requires loader `frameskip=1` (the training entrypoint already uses
this). The encoder's repeated images for its two-frame tubelet do not change the
physical grid. Multiple camera views use their own native clocks and mappings;
this is not a cross-camera fusion/synchronization implementation.

Plans are cached per worker (at most128 streams), keyed by paths, clock parameters,
frame count, and timestamp-file/trajectory size and modification time. NumPy's
existing worker-seeded RNG selects uniformly among admissible frame anchors.
Short or corrupt video retries are bounded; clock/schema errors propagate.

Changing the data clock changes the training distribution. This implementation
does not establish the historical data path of released checkpoints or improved
robot performance. Preserve `legacy` when reproducing an experiment known to use
that convention, and record the choice explicitly.
