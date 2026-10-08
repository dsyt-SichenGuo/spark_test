"""Unified high-level launcher for G1 walking in static reconstructed worlds.

Like run_unitree_g1_benchmark.py's Isaac tensor branch, this entry point
configures an experiment and launches an isolated Isaac runtime. The existing
spark_pipeline.autonomy.unitree_g1_environment_tensor_walk owns the scene, task,
tensor control loop, WBT/Sport inference, and per-environment resets. It uses
static Marble or GPT-v4 assets without people, animation, or replay cameras.
The marble-full option adds original SPZ appearance over the static collider.

Examples (run in the same Isaac Lab environment as the tensor example)::

    python scrpits/run_unitree_g1_usd_environment_walk_benchmark.py \
        --environment marble-full

    python scrpits/run_unitree_g1_usd_environment_walk_benchmark.py \
        --environment gpt-v4 --robot-position-mode random
    python scrpits/run_unitree_g1_usd_environment_walk_benchmark.py \
        --environment marble --robot-position-mode manual --robot-yaw-mode manual \
        --robot-position -0.2 -1.0 --robot-yaw-degrees 217.24
    python scrpits/run_unitree_g1_usd_environment_walk_benchmark.py \
        --environment gpt-v4 --goal-position-mode manual --goal-position -4 -4

Additional Isaac AppLauncher options, e.g. --experience, are passed through.
The historical wbtsafe/sportsafe names select the original runtime's WBT/Sport
controllers; they do not add a safety filter to the original task.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME = REPO_ROOT / "pipeline/spark_pipeline/autonomy/unitree_g1_environment_tensor_walk.py"
DEFAULT_WORLD = REPO_ROOT / "reconstructions" / "marble-1.1-static"
# One registry contains all asset-specific alignment and legacy spawn defaults.
ENVIRONMENT_ASSETS = {
    "marble": {
        "world_dir": DEFAULT_WORLD,
        "world_z_offset": 1.02,
        "spawn_base_z": 0.793 - 1.50 + 1.02,
        "robot_position": (-0.2, -1.0),
        "robot_yaw_degrees": math.degrees(0.65 + math.pi),
        "light_intensity": 550.0,
        "terrain_mesh": "/World/MarbleCollider",
    },
    "gpt-v4": {
        "world_dir": REPO_ROOT / "reconstructions/gpt-v4-static",
        "world_z_offset": 1.47,
        "spawn_base_z": 0.793,
        "robot_position": (-4.0, -4.0),
        "robot_yaw_degrees": 0.0,
        "light_intensity": 400.0,
        "terrain_mesh": None,
    },
}
# Reuse the identical static collider; appearance comes from the original SPZ.
MARBLE_FULL_PACKAGE = REPO_ROOT / "reconstructions/marble-1.1-supermarket-full-appearance-1s-18s"
ENVIRONMENT_ASSETS["marble-full"] = {
    **ENVIRONMENT_ASSETS["marble"],
    "gaussian_dir": MARBLE_FULL_PACKAGE / "assets/marble_1_1_people_1s_18s",
    "gaussian_alignment": MARBLE_FULL_PACKAGE / "outputs/marble-people-1s-18s/alignment.json",
}
GAUSSIAN_FILES = {"full": "world_full_res.spz", "500k": "world_500k.spz",
                  "150k": "world_150k.spz", "100k": "world_100k.spz"}

POLICY_RUNTIME = {
    "UnitreeG1WBTSafePolicy": "wbtsafe",
    "UnitreeG1SportSafePolicy": "sportsafe",
}

# Keep the runtime's task defaults and forward every task option explicitly.
INTEGER_OPTIONS = {
    "num_envs": 1,
    "steps": -1,
    "episode_steps": 12000,
    "render_every": 5,
    "warmup_steps": 20,
    "transition_steps": 50,
    "stall_window_steps": 150,
}
FLOAT_OPTIONS = {
    "env_spacing": 20.0,
    "wall_margin": 5.0,
    "min_goal_distance": 3.0,
    "goal_tolerance": 0.45,
    "goal_yaw_tolerance_degrees": 15.0,
    "fall_height": 0.22,
    "fall_tilt_degrees": 55.0,
    "max_forward_speed": 0.12,
    "max_lateral_speed": 0.10,
    "max_yaw_rate": 0.30,
    "camera_distance": 3.0,
    "camera_height": 2.0,
    "camera_lookahead": 1.5,
    "camera_follow_smoothing": .35,
    "camera_clearance": .18,
    "interior_light_intensity": 1600.0,
    "camera_light_intensity": 1000.0,
    "interior_ambient_intensity": .8,
}


def build_parser(*, for_runtime: bool = False) -> argparse.ArgumentParser:
    """Expose experiment configuration without importing or starting Isaac."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--policy", choices=tuple(POLICY_RUNTIME.values()))
    policy.add_argument("--policy-config", choices=tuple(POLICY_RUNTIME),
                        help="registered policy name, as in the G1 benchmark")
    for name, default in INTEGER_OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in FLOAT_OPTIONS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    parser.add_argument("--environment", choices=tuple(ENVIRONMENT_ASSETS), default="marble")
    parser.add_argument("--camera-mode", choices=("fixed", "follow"), default="fixed",
                        help="fixed episode view (default), or stable robot-following view")
    parser.add_argument("--record-video", action="store_true",
                        help="record MP4 views; marble-full records both mesh and GS")
    parser.add_argument("--video-dir", type=Path, default=Path("videos"), help="MP4 output directory (default: ./videos)")
    parser.add_argument("--video-fps", type=float, default=None,
                        help="playback FPS; default: 50 / render-every, matching simulation time")
    parser.add_argument("--gaussian-quality", choices=tuple(GAUSSIAN_FILES), default="full",
                        help="marble-full appearance resolution; default: all 1.92M Gaussians")
    parser.add_argument("--gaussian-resolution", nargs=2, type=int, default=(960, 720),
                        metavar=("WIDTH", "HEIGHT"), help="composite image size")
    parser.add_argument("--gaussian-snapshot", type=Path,
                        help="save the final composite PNG; also enables rendering when headless")
    parser.add_argument("--world-dir", type=Path, default=None,
                        help="override this environment's static asset directory")
    parser.add_argument("--init-mode", choices=("random", "fixed-start", "fixed-goal"),
                        help="legacy shorthand; prefer the three independent mode options")
    for name in ("robot-position", "robot-yaw", "goal-position"):
        parser.add_argument(f"--{name}-mode", choices=("random", "manual"),
                            help="default: random; supplying the corresponding value selects manual")
    parser.add_argument("--robot-position", nargs=2, type=float, metavar=("X", "Y"),
                        help="manual initial XY in asset-local coordinates (meters)")
    parser.add_argument("--robot-yaw-degrees", type=float,
                        help="manual initial heading in degrees, +X=0, +Y=90")
    parser.add_argument("--goal-position", nargs=2, type=float, metavar=("X", "Y"),
                        help="manual target XY in asset-local coordinates (meters)")
    parser.add_argument("--sampling-bounds", nargs=4, type=float,
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="optional random sampling rectangle inside the scene bounds")
    parser.add_argument("--spawn-height", type=float, default=None,
                        help="optional base Z in the aligned environment's coordinates")
    parser.add_argument("--seed", type=int, default=None)
    if not for_runtime:
        parser.add_argument("--isaac-device", "--device", dest="device", default=None,
                            help="Isaac device; omitted uses AppLauncher's default")
    if not for_runtime:
        parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate assets and print the command without starting Isaac")
    return parser


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Reject invalid experiments and missing scene assets before Kit starts."""
    if args.steps == 0 or args.steps < -1:
        parser.error("--steps must be positive or -1")
    if args.video_fps is not None and (not math.isfinite(args.video_fps) or args.video_fps <= 0):
        parser.error("--video-fps must be finite and positive")
    for name in ("num_envs", "episode_steps", "render_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.stall_window_steps < 2:
        parser.error("--stall-window-steps must be at least 2")
    for name in ("warmup_steps", "transition_steps"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    for name in FLOAT_OPTIONS:
        value = getattr(args, name)
        zero_allowed = name in ("wall_margin", "camera_lookahead", "interior_light_intensity",
                                "camera_light_intensity", "interior_ambient_intensity")
        if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
            parser.error(f"--{name.replace('_', '-')} must be finite and "
                         + ("nonnegative" if zero_allowed else "positive"))
    for name in ("goal_yaw_tolerance_degrees", "fall_tilt_degrees"):
        if getattr(args, name) > 180:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 180]")
    asset = ENVIRONMENT_ASSETS[args.environment]
    if any(v < 64 or v > 1920 for v in args.gaussian_resolution):
        parser.error("--gaussian-resolution dimensions must be in [64, 1920]")
    if args.gaussian_snapshot is not None and "gaussian_dir" not in asset:
        parser.error("--gaussian-snapshot requires --environment marble-full")
    if "gaussian_dir" in asset:
        for path in (asset["gaussian_dir"] / GAUSSIAN_FILES[args.gaussian_quality],
                     asset["gaussian_alignment"]):
            if not path.is_file():
                parser.error(f"Gaussian asset not found: {path}")
    legacy_modes = {
        "random": ("random", "random", "random"),
        "fixed-start": ("manual", "manual", "random"),
        "fixed-goal": ("random", "random", "manual"),
    }
    for index, (mode_name, value_name) in enumerate((
        ("robot_position_mode", "robot_position"),
        ("robot_yaw_mode", "robot_yaw_degrees"),
        ("goal_position_mode", "goal_position"),
    )):
        mode, value = getattr(args, mode_name), getattr(args, value_name)
        if args.init_mode is not None:
            legacy = legacy_modes[args.init_mode][index]
            if mode is not None and mode != legacy:
                parser.error(f"--{mode_name.replace('_', '-')} conflicts with --init-mode")
            mode = legacy
            # Preserve historical fixed-start defaults for legacy commands only.
            if mode == "manual" and value is None and value_name != "goal_position":
                value = asset[value_name]
                setattr(args, value_name, value)
        mode = mode or ("manual" if value is not None else "random")
        if mode == "manual" and value is None:
            parser.error(f"--{mode_name.replace('_', '-')} manual requires --{value_name.replace('_', '-')}")
        if mode == "random" and value is not None:
            parser.error(f"--{value_name.replace('_', '-')} conflicts with random mode")
        setattr(args, mode_name, mode)
    for name in ("robot_position", "goal_position", "sampling_bounds"):
        value = getattr(args, name)
        if value is not None and not all(math.isfinite(v) for v in value):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if args.robot_yaw_degrees is not None and not math.isfinite(args.robot_yaw_degrees):
        parser.error("--robot-yaw-degrees must be finite")
    if args.spawn_height is not None and (not math.isfinite(args.spawn_height) or args.spawn_height <= 0):
        parser.error("--spawn-height must be finite and positive")
    args.world_dir = (args.world_dir or asset["world_dir"]).expanduser().resolve()
    if not RUNTIME.is_file():
        parser.error(f"tensor runtime not found: {RUNTIME}")
    try:
        manifest = json.loads((args.world_dir / "environment.json").read_text())
        usd_path = (args.world_dir / manifest["usd_path"]).resolve()
        if not usd_path.is_file():
            raise ValueError(f"environment USD not found: {usd_path}")
        if manifest["up_axis"] != "Z" or manifest["meters_per_unit"] != 1.0:
            raise ValueError("environment must be Z-up and use meters")
        if manifest.get("animation", {}).get("has_people_animation") or manifest.get("people_prim"):
            raise ValueError("use a static asset without people or baked animation")
        inventory_path = args.world_dir / "inspection" / "prim_inventory.json"
        inventory = json.loads(inventory_path.read_text())
        world = next(record for record in inventory if record.get("path") == manifest["static_environment_prim"])
        bounds = world["world_bounds_default"]
        args.camera_bounds_min = [float(v) for v in bounds["min"]]
        args.camera_bounds_max = [float(v) for v in bounds["max"]]
        args.camera_bounds_min[2] += asset["world_z_offset"]
        args.camera_bounds_max[2] += asset["world_z_offset"]
        low = [float(bounds["min"][axis]) + args.wall_margin for axis in (0, 1)]
        high = [float(bounds["max"][axis]) - args.wall_margin for axis in (0, 1)]
        if not all(math.isfinite(v) for v in (*low, *high)):
            raise ValueError("environment XY bounds must be finite")
        if args.sampling_bounds is not None:
            xmin, xmax, ymin, ymax = args.sampling_bounds
            low, high = [xmin, ymin], [xmax, ymax]
            if any(low[i] < bounds["min"][i] or high[i] > bounds["max"][i] for i in (0, 1)):
                raise ValueError("--sampling-bounds must lie inside the environment XY bounds")
        if any(low[axis] >= high[axis] for axis in (0, 1)):
            raise ValueError("wall margin or sampling bounds leaves no usable sampling area")
        fixed_start = args.robot_position if args.robot_position_mode == "manual" else None
        fixed_goal = args.goal_position if args.goal_position_mode == "manual" else None
        for fixed in (fixed_start, fixed_goal):
            if fixed is not None and any(fixed[i] < bounds["min"][i] or fixed[i] > bounds["max"][i] for i in (0, 1)):
                raise ValueError("manual position must lie inside the environment XY bounds")
        if fixed_start is not None and fixed_goal is not None:
            if math.dist(fixed_start, fixed_goal) < args.min_goal_distance:
                raise ValueError("manual start and goal must be at least --min-goal-distance apart")
        else:
            fixed = fixed_start if fixed_start is not None else fixed_goal
            farthest = (math.hypot(*(max(abs(low[i] - fixed[i]), abs(high[i] - fixed[i])) for i in (0, 1)))
                        if fixed is not None else math.hypot(high[0] - low[0], high[1] - low[1]))
            if args.min_goal_distance >= farthest:
                raise ValueError("--min-goal-distance exceeds the available sampling area")
        args.xy_min, args.xy_max = low, high
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as exc:
        parser.error(f"invalid static environment configuration: {exc}")


def build_runtime_command(args, launcher_args=()) -> list[str]:
    """Translate benchmark-style configuration to the unchanged tensor task."""
    policy = POLICY_RUNTIME[args.policy_config] if args.policy_config else (args.policy or "wbtsafe")
    command = [sys.executable, str(RUNTIME), "--policy", policy]
    for name in (*INTEGER_OPTIONS, *FLOAT_OPTIONS, "environment", "robot_position_mode", "robot_yaw_mode", "goal_position_mode", "world_dir",
                 "robot_yaw_degrees", "spawn_height", "seed", "device", "gaussian_quality", "gaussian_snapshot", "camera_mode", "video_dir", "video_fps"):
        value = getattr(args, name)
        if value is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    for name in ("robot_position", "goal_position", "sampling_bounds", "gaussian_resolution"):
        values = getattr(args, name)
        if values is not None:
            command.extend(("--" + name.replace("_", "-"), *(str(v) for v in values)))
    if args.headless:
        command.append("--headless")
    if args.record_video:
        command.append("--record-video")
    command.append("--real-time" if args.real_time else "--no-real-time")
    command.extend(launcher_args)
    return command


def run(args, launcher_args=()) -> subprocess.CompletedProcess | None:
    """Launch with the current Python environment, propagating runtime failures."""
    command = build_runtime_command(args, launcher_args)
    print(f"[Environment high level] {shlex.join(command)}", flush=True)
    if args.dry_run:
        return None
    if args.record_video:
        # subprocess.run kills its child on KeyboardInterrupt. Give the recorder
        # a chance to write the MP4 index before Isaac shuts down instead.
        process = subprocess.Popen(command, start_new_session=(os.name == "posix"))
        interrupted = False
        while True:
            try:
                code = process.wait()
                break
            except KeyboardInterrupt:
                if not interrupted and process.poll() is None:
                    interrupted = True
                    print("[Video] stopping runtime; waiting for MP4 finalization...", flush=True)
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGINT)
                    else:
                        process.send_signal(signal.SIGINT)
        if interrupted:
            raise KeyboardInterrupt
        if code:
            raise subprocess.CalledProcessError(code, command)
        return subprocess.CompletedProcess(command, code)
    return subprocess.run(command, check=True)


def main() -> None:
    parser = build_parser()
    args, launcher_args = parser.parse_known_args()
    validate_args(args, parser)
    try:
        run(args, launcher_args)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
