#!/usr/bin/env python3
"""Independent WBT/Sport keyboard walking with synchronized mesh/GS windows."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reconstruction_walk.config import DEFAULT_CONFIG


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--policy", choices=("wbt", "sport"), default="wbt")
    p.add_argument("--physics-device", default="cpu")
    p.add_argument("--render-device", default="cuda:0")
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--headless", action="store_true", help="Physics only; no visual capture or GS worker")
    p.add_argument("--no-gs", action="store_true", help="Mesh-only interactive diagnostic")
    p.add_argument("--prepare-only", action="store_true", help="Prepare and validate metric assets without Isaac/GPU")
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Prepare assets and print selection without starting Isaac")
    p.add_argument("--policy-smoke-test", action="store_true", help="CPU test of actual selected policy weights; no physics")
    p.add_argument("--gaussian-smoke-test", action="store_true", help="GPU render of one registered camera; no Isaac")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--render-every", type=int, default=5)
    p.add_argument("--light-scale", type=float, default=None)
    p.add_argument("--mesh-color", action=argparse.BooleanOptionalAction, default=False,
                   help="Show source mesh colors; default is neutral gray geometry (also switchable in the mesh window)")
    p.add_argument("--record", action="store_true", help="Start MP4 recording when the first GS/mesh pair is displayed")
    p.add_argument("--video-dir", type=Path, default=None, help="MP4 output directory; default: selected scene cache/videos")
    p.add_argument("--video-fps", type=float, default=30., help="MP4 wall-clock frame rate (default: 30)")
    p.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scripted-command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                   help="Bounded test command after standing warmup; omit for keyboard control")
    p.add_argument("--max-resets", type=int, default=-1, help="-1 permits unlimited nearby fall recovery")
    p.add_argument("--test-fall-at", type=int, default=None,
                   help="Diagnostic only: physically tip the robot once at this control step")
    return p


def policy_smoke_test(name):
    import numpy as np
    import torch
    from spark_robot import UnitreeG1WholeBodyDynamic1Config
    from reconstruction_walk.policies import LocomotionController
    torch.set_num_threads(1)
    controller = LocomotionController(name, UnitreeG1WholeBodyDynamic1Config(), "cpu")
    feedback = {"body_joint_pos": torch.zeros(1, 29), "body_joint_vel": torch.zeros(1, 29),
                "root_pose_w": torch.tensor([[0., 0., .8, 0., 0., 0., 1.]]),
                "root_angular_velocity_b": torch.zeros(1, 3)}
    upper = torch.zeros(1, 17)
    controller.reset()
    first, _ = controller.infer(feedback, np.zeros(3), upper)
    for i in range(100):
        target, info = controller.infer(feedback, np.array([.05, 0., .05]) if i > 70 else np.zeros(3), upper)
        assert target.shape == (1, 29) and torch.isfinite(target).all()
        assert "target_actuated_pos" in info
    controller.reset()
    again, _ = controller.infer(feedback, np.zeros(3), upper)
    assert torch.allclose(first, again, atol=1e-6), "Policy hidden state reset is not reproducible"
    print(json.dumps({"policy": name, "actual_weight_inference": "passed", "inference_steps": 102,
                      "output_joints": 29, "state_reset": "passed", "physics_test": False}))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.steps == 0 or args.steps < -1 or args.render_every < 1:
        p.error("--steps must be positive or -1 and --render-every must be positive")
    if not 64 <= args.width <= 1280 or not 64 <= args.height <= 960:
        p.error("render dimensions must fit 64–1280 × 64–960")
    if args.light_scale is not None and not 0 <= args.light_scale <= 8:
        p.error("--light-scale must be in [0, 8]")
    if not math.isfinite(args.video_fps) or not 1 <= args.video_fps <= 120:
        p.error("--video-fps must be finite and in [1, 120]")
    if args.record and (args.headless or args.no_gs):
        p.error("--record requires the GS/mesh views; remove --headless and --no-gs")
    if args.max_resets < -1 or (args.test_fall_at is not None and args.test_fall_at < 0):
        p.error("--max-resets must be -1 or nonnegative; --test-fall-at must be nonnegative")
    if args.scripted_command is not None and not all(math.isfinite(x) for x in args.scripted_command):
        p.error("--scripted-command must contain finite values")
    if args.policy_smoke_test:
        policy_smoke_test(args.policy)
        return
    # A cache rebuild imports standalone pxr for USD export. Isaac must import
    # its own USD libraries first, so preparation must not run in this process.
    from reconstruction_walk.config import SceneConfig
    command = [sys.executable, "-m", "reconstruction_walk.prepare", "--scene", str(args.scene.resolve())]
    if args.force_prepare:
        command.append("--force")
    subprocess.run(command, cwd=ROOT, check=True)
    cfg = SceneConfig.read(args.scene)
    report = json.loads((cfg.cache / "prepared.json").read_text())
    if args.prepare_only or args.dry_run:
        print(json.dumps({"policy": args.policy, "scene": str(cfg.path), "cache": str(cfg.cache),
                          "spawn": report["initial_spawn_xy_floor_yaw"], "backend": cfg.data["gaussian"]["backend"],
                          "mesh_color": args.mesh_color,
                          "light_scale": args.light_scale if args.light_scale is not None else cfg.data.get("light_scale", 2)}, indent=2))
        return
    if args.gaussian_smoke_test:
        import numpy as np
        from PIL import Image
        from reconstruction_walk.gaussian_process import GaussianProcess
        c = json.loads(cfg.asset(cfg.data["cameras"]).read_text())[0]
        a = report["gaussian_to_world"]
        r, t, s = np.array(a["rotation"]), np.array(a["translation"]), a["scale"]
        camera = np.eye(4)
        camera[:3, :3] = r @ np.array(c["rotation"])
        camera[:3, 3] = s * r @ np.array(c["position"]) + t
        k = [[c["fx"] * args.width / c["width"], 0, args.width / 2],
             [0, c["fy"] * args.height / c["height"], args.height / 2], [0, 0, 1]]
        process = GaussianProcess(cfg, args.render_device)
        try:
            rgba, depth = process.render(7, camera, k, args.width, args.height)
            assert rgba.max() > 0 and np.any(depth > 0) and np.isfinite(depth).all()
            Image.fromarray(rgba).save(cfg.cache / "gaussian-check.png")
            print("[test] Actual GPU render, world/native camera transform, depth and state-ID protocol passed")
        finally:
            process.close()
        return
    # Construct the external renderer before starting Kit. No CUDA extensions
    # are imported into the Isaac host by this launcher.
    worker = app = pipeline = None
    exit_code = 0
    try:
        if not args.headless and not args.no_gs:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is unavailable in this session. Use a GPU-enabled spark_isaac terminal; "
                                   "--prepare-only and --policy-smoke-test can run without a GPU.")
            from reconstruction_walk.gaussian_process import GaussianProcess
            worker = GaussianProcess(cfg, args.render_device)
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": args.headless, "hide_ui": args.headless,
            "width": args.width * 2, "height": args.height + 120,
            "renderer": "RaytracedLighting", "anti_aliasing": 0,
            "limit_cpu_threads": 8,
            "extra_args": ["--/exts/omni.kit.hotkeys.core/hotkeys_enabled=false"]})
        from reconstruction_walk.runtime import ReconstructionWalkPipeline
        pipeline = ReconstructionWalkPipeline(cfg, args, app, worker)
        pipeline.run()
    except KeyboardInterrupt:
        print("[walk] Interrupted; closing simulator and Gaussian worker", flush=True)
    except Exception:
        # Kit shutdown may terminate the interpreter before an outer traceback
        # is printed. Record the failure before entering its close routine.
        import traceback
        exit_code = 1
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            if worker is not None:
                worker.close()
            if app is not None:
                app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    main()
