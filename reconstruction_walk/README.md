# G1 Keyboard Walking in Reconstructed Environments

This implementation provides WBT/Sport keyboard walking with synchronized mesh and Gaussian windows. Launchers are under `scrpits/`, and the new components are in this directory. It reuses the project's Unitree G1 Isaac agent and existing policy weights without modifying the original teleoperation, benchmark, agent, or policy implementation. PhysX handles physical contact.

## Launch

```bash
conda activate spark_isaac
cd /home/gsc/research/spark_test

# WBT
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy wbt

# Sport: close the previous instance before launching another.
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy sport
```

You can also run `python scrpits/run_unitree_g1_reconstruction_teleop.py --policy wbt`. The shell launcher uses Python from the active environment, or the interpreter specified by `SPARK_PYTHON`.

The current default in `config.py` is `reconstructions/office-hallway-matcha/configs/new_ratio1.json`. Scene configuration files are maintained only under that asset directory; the duplicate JSON previously in this package has been removed.

| Configuration | Gaussian model | Mesh | Cameras |
| --- | --- | --- | --- |
| `old.json` | `old/gaussian/point_cloud.ply` | `old/meshes/default.ply` | `old/cameras.json` |
| `new.json` | `new/gaussian/point_cloud.ply` | `new/meshes/default.ply` | `new/cameras.json` |
| `new_ratio1.json` | `new/gaussian/point_cloud.ply` | `new/meshes/ratio1.ply` | `new/cameras.json` |

Use `--scene` to select a configuration explicitly:

```bash
./scrpits/launch_unitree_g1_reconstruction_walk.sh \
  --scene reconstructions/office-hallway-matcha/configs/old.json --policy wbt
```

Paths in a scene JSON are relative to that JSON's directory. See the [asset README](../reconstructions/office-hallway-matcha/README.md) for the complete layout and configuration choices. The original source assets are retained.

Two windows display the mesh environment with G1 and the Gaussian environment with the same G1. Their images use one captured robot state and camera and are published together. The Gaussian view composites the robot with depth occlusion. The environment is static, the camera follows the robot, and execution is not limited by the training video's duration.

## Mesh Appearance and Lighting

The mesh defaults to neutral gray to expose surface shape. Enable the original vertex colors with `--mesh-color`, or select **Mesh source colors** at the top of the mesh window. Clear the checkbox to return to gray.

```bash
# Gray geometry: the default.
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy wbt

# Original mesh colors; also supported with Sport.
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy wbt --mesh-color

# Explicitly disable source colors.
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy sport --no-mesh-color
```

The appearance switch changes only the environment mesh material. Robot appearance, collision, Gaussian appearance, and camera synchronization remain unchanged. Overrides are authored in the USD session layer, so switching does not rewrite cached USDs or source PLYs and does not require cache preparation.

Lighting defaults to a coefficient of **2**, including ambient lighting, a dome, camera lights, and mesh-only interior fill lights along the reconstruction path. Override it with `--light-scale`. Native lights do not change the trained Gaussian background.

## Controls and Recovery

| Key | Action |
| --- | --- |
| W / S | Forward / backward in the robot frame |
| A / D | Turn left / right |
| Q / E | Strafe left / right |
| Space | Clear input and command velocity |
| R | Reinitialize near the current location |
| F9 | Start / stop the combined GS/mesh MP4 recording |
| Ctrl+C or application close | Exit |

Commands are bounded and acceleration-limited. Releasing keys reduces the command smoothly; losing application focus clears input. WBT uses its standing expert while idle and blends toward locomotion with movement input. After stopping, it blends back toward standing once speed decreases. Sport uses its own recurrent policy and actuation configuration. Initialization and recovery include approximately 1.4 seconds of simulation warmup before accepting movement commands. A zero velocity command does not lock the robot's physical position.

The input callback accepts enum and string key representations and filters character/repeat events before reading their input. This prevents character events from causing an AttributeError or triggering extra movement and reset actions.

A persistent low base height or excessive tilt confirms a fall. Recovery first chooses the nearest prevalidated spawn within 2 m. If none is nearby, it selects a candidate near the last stable position, with the initial candidate as the final fallback. Robot pose and velocity, policy hidden state, keyboard input, and the follow camera are reset without restarting the simulator. Nonfinite robot poses and falling below the environment also trigger recovery.

## MP4 Recording

Recording is off by default. In the Gaussian window, click **Start recording (F9)**, then **Stop recording (F9)** when finished. F9 performs the same action while the application is focused. A request made before the first synchronized pair waits for that pair; pressing again cancels the request.

Each recording interval produces one MP4 containing **GS on the left and mesh on the right**, including the same simulated robot in both halves. The output uses the rendered images rather than the window borders, buttons, or desktop. At the default 640 × 480 view size, the video is 1280 × 480. Odd dimensions are padded at the outer edge for encoding.

Files are named `gs_mesh_<UTC timestamp>.mp4` and saved under the selected scene's cache `videos/` directory. For the current default configuration, this is `log/reconstruction_walk/office-hallway-matcha/new_ratio1/videos/`. The status label and console report the saved path. Stopping the application through normal shutdown or Ctrl+C finalizes an active MP4 automatically. Each subsequent start creates a separate file.

```bash
# Choose the output folder and FPS, then start/stop manually with the button or F9.
./scrpits/launch_unitree_g1_reconstruction_walk.sh \
  --video-dir log/reconstruction_walk/recordings --video-fps 30

# Optionally begin recording automatically when the first matching pair appears.
./scrpits/launch_unitree_g1_reconstruction_walk.sh --policy wbt --record
```

The default recording rate is 30 FPS, measured in wall time. A background encoder samples the latest displayed pair and repeats it between render updates, so slow rendering does not accelerate playback. Both halves always come from the same published state. Frame repetition does not increase the underlying rendering rate. Encoding retains a bounded amount of image memory instead of accumulating a recording in RAM. OpenCV in `spark_isaac` supplies the MP4 encoder. Recording requires both views and is unavailable with `--headless` or `--no-gs`.

## Scene Preparation and Scale

Each packaged configuration uses a separate cache under `log/reconstruction_walk/office-hallway-matcha/old/`, `new/`, or `new_ratio1/`:

| File | Contents |
| --- | --- |
| `visual.usdc` | Complete display mesh, original colors/material, and world transform |
| `collision.usdc`, `collision.npz` | Simplified static collision geometry and spatial-query data |
| `spawns.npy` | Candidates checked for floor support, height variation, wall clearance, and overhead clearance |
| `alignment.json`, `prepared.json` | Coordinate transforms, scale provenance, source signatures, and preparation results |

The collision mesh merges nearby vertices and retains surviving surface triangles. It does not randomly remove triangles or add a hidden infinite support plane. Indexed triangle-ray queries handle spawn checks and floor queries without scanning the full display mesh every frame. Simplification can change small holes and details.

**Metric scale is estimated**, using the fitted floor plane and an assumed median camera height of **1.55 m**. The new reconstruction is approximately **1.36865 m per reconstruction unit**; the old reconstruction is approximately **1.32698 m per reconstruction unit**. These are not measured calibrations. Set `meters_per_unit` when a measured scale is available, or change `camera_height_m` when the actual recording-camera height is known. Changes invalidate the cache; `--force-prepare` also requests a rebuild.

The display mesh uses a USD root transform. Gaussian rendering applies the inverse camera transform, and depths are converted to simulation-world meters. Source mesh and Gaussian files are preserved.

Preparation runs in an independent Python process. This prevents standalone PXR libraries imported during USD export from conflicting with Isaac's own USD libraries at startup. Replacing a mesh or using `--force-prepare` therefore does not require a separate manual preparation run.

## Options and Diagnostics

```bash
# CPU scene preparation and geometry checks only.
python scrpits/run_unitree_g1_reconstruction_teleop.py --prepare-only

# Prepare assets and print the selected configuration without starting Isaac.
python scrpits/run_unitree_g1_reconstruction_teleop.py --dry-run

# Real policy weights and hidden-state reset checks; no physics.
python scrpits/run_unitree_g1_reconstruction_teleop.py --policy wbt --policy-smoke-test
python scrpits/run_unitree_g1_reconstruction_teleop.py --policy sport --policy-smoke-test

# Actual CUDA Gaussian rendering with camera, depth, and response-ID checks.
python scrpits/run_unitree_g1_reconstruction_teleop.py --gaussian-smoke-test

# A finite physics-only run; headless mode does not create the two views.
python scrpits/run_unitree_g1_reconstruction_teleop.py --policy sport \
  --headless --steps 500 --scripted-command 0.06 0 0 --no-real-time

# Deliberately inject one physical fall and check nearby recovery.
python scrpits/run_unitree_g1_reconstruction_teleop.py --policy wbt \
  --headless --steps 500 --scripted-command 0.06 0 0 \
  --test-fall-at 150 --max-resets 3 --no-real-time

# Adjust rendering load and brightness.
python scrpits/run_unitree_g1_reconstruction_teleop.py --policy wbt \
  --width 640 --height 480 --render-every 10 --light-scale 2

# CPU regression checks.
python -m pytest -q tests/test_reconstruction_walk.py
```

Use `--no-gs` for an interactive mesh-only diagnostic. `--scripted-command` and `--test-fall-at` are diagnostic options; omit them for keyboard operation.

The physics step is 5 ms, with one control update every four steps: a 50 Hz simulation control period. By default, rendering is submitted every five control updates. Gaussian rendering runs in an independent CUDA process with at most one outstanding request. Both windows use the same state snapshot. RTX capture saves and restores the robot state without intentionally pausing the control timeline. Displayed frames can lag the latest simulated state while rendering completes.

Actual wall-clock performance depends on mesh size, resolution, hardware, and drivers; a 50 Hz simulation control period does not guarantee 50 Hz real-time execution. Logs report effective throughput every 250 steps. Increase `--render-every` or reduce image dimensions if rendering is too slow.

The default single-robot PhysX simulation and policy inference use CPU; Gaussian rendering uses CUDA. Override with `--physics-device cuda:0` and `--render-device cuda:0` as appropriate. WBT and Sport keep their own actuation settings and are selected at startup. The task applies command limits and physical collision; it does not add the project's predictive safety filter.

Generated outputs include `walk-mesh-<policy>.png`, `walk-gaussian-<policy>.png`, `run-<policy>.json`, and `gaussian-worker.log` in the selected cache. Normal shutdown closes the Gaussian worker.

## Supporting Other Gaussian Methods

Copy a scene JSON and select it with `--scene /path/to/your_scene.json`. Source assets and generated caches remain separate.

When changing training versions, match **`gaussian.path`, `mesh.path`, and `cameras`**. They must describe the same reconstruction frame or have the necessary explicit transform. The same input video can produce different rotation, translation, and scale across training runs. For example, `old/gaussian/point_cloud.ply` must use `old/cameras.json`, while the new model uses `new/cameras.json`.

| Backend | Input |
| --- | --- |
| `matcha_2dgs` | MAtCha 2DGS PLY with a matching `renderer_source` |
| `gsplat_3dgs` | Standard Graphdeco 3DGS PLY with three scale fields, quaternion, opacity, and SH coefficients |
| `your_module:YourRenderer` | A custom adapter following the interface in `backends.py` |

A `.ply` extension does not make these formats interchangeable. Register another method with `register_backend()`, or provide `module:Class`. The constructor receives `(SceneConfig, gaussian_to_world_alignment, device)`. Implement `count`, `render(camera_c2w_cv, K, width, height)`, and `close()`. Rendering must return RGBA uint8 and **camera-Z depth in world units**, not Euclidean ray distance. The compositor requires depth.

Mesh input is triangular PLY with optional RGB vertex colors. Missing colors use gray. Convert other mesh formats to triangular PLY first. Camera JSON is a list containing `position`, `rotation` in the CV camera-to-world convention (+Z forward, +Y down), `fx`, `fy`, `width`, and `height`. It supports floor orientation, scale estimation, and spawn selection. Normalize other camera conventions before use.

`gaussian.to_mesh` optionally specifies a Gaussian-native to mesh-native similarity transform: positive `scale`, a 3 × 3 `rotation`, and a three-component `translation`. A missing value assumes the same native frame. The complete mesh is then calibrated into the physical world, so another rendering method does not require rewriting keyboard or locomotion control.

## Module Responsibilities

| Files | Responsibility |
| --- | --- |
| `config.py`, scene JSONs | Asset paths, renderer selection, cache locations, and calibration settings |
| `prepare.py`, `geometry.py` | Floor calibration, display/collision USDs, spatial queries, and spawn candidates |
| `task.py` | Keyboard intent, velocity limits, fall detection, and local recovery selection |
| `policies.py` | WBT/Sport inference and reset adapters |
| `runtime.py` | Agent, environment, control loop, and keyboard events |
| `appearance.py` | Session-only gray/source-color mesh material selection |
| `backends.py`, `worker.py`, `gaussian_process.py` | Replaceable Gaussian rendering and process lifecycle |
| `rendering.py` | Matching-state views, instance filtering, depth occlusion, and lighting |
| `recording.py` | Manual recording intervals, wall-time pacing, and combined GS-left/mesh-right MP4 encoding |

The new environment and task are composed explicitly. The implementation does not change the original `SparkEnvironment` reset contract; it initializes the robot through the agent's `root_pose_w` interface.

## Local Validation

The following checks were completed in `spark_isaac` with the host GPU where needed:

- 12 CPU test cases in `test_reconstruction_walk.py` covering calibration, support queries, held-key release and enum/string input, F9 toggling, fall confirmation, nearby recovery, depth composition, float32 camera conversion, and invalid configuration handling.
- Three tests in `test_reconstruction_recording.py` exercising actual MP4 encoding/decoding, left/right color order, wall-time duration without fresh renders, repeated starts/stops, delayed start until a paired frame, and rejection of mismatched inputs.
- Actual GS/mesh GPU recording at 1280 × 480, plus two intervals triggered through the renderer's recording controls. All MP4s decoded successfully, including the clip finalized automatically on application exit. Logs, a sample MP4, and results are under `log/reconstruction_walk/recording-validation/`.
- Real WBT/Sport weight inference and reproducible hidden-state resets.
- A 320-step WBT run with both views, an injected physical fall, nearby recovery, and continued walking.
- A 160-step Sport run with both views, plus a 350-step physics run with an injected fall and recovery.
- Actual MAtCha CUDA rendering, camera transforms, depth, and state identifiers, with visual inspection of mesh/GS images.
- An 80-step run with the ratio1 mesh after forced cache preparation, and an 80-step run with the old assets that included one fall and automatic recovery.
- Preparation of all three packaged configurations and actual CUDA loading/rendering of both copied Gaussian versions through the copied renderer.
- A synthetic standard-3DGS CUDA rendering check for color, alpha, and camera-Z depth. Another real 3DGS scene has not yet received an end-to-end walking test.
- The original 52 code files checked at the start of implementation retained their SHA256 values.

Package results are in `reconstructions/office-hallway-matcha/validation.json` and `validation/`; `manifest.json` records copy provenance and hashes. Initial runtime validation records and `test-*.log` files are also retained under `log/reconstruction_walk/office-hallway-matcha/`. These short checks validate integration, basic contact, rendering, and recovery; they do not establish reliable walking over every part of the reconstructed floor.
