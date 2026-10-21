"""Authoritative G1 environment and keyboard walking pipeline."""

import json
import math
import time

import numpy as np

from .geometry import SurfaceQueries
from .policies import LocomotionController
from .task import KeyboardWalkTask


class ReconstructionEnvironment:
    def __init__(self, cfg, args, app):
        import torch
        import omni.usd
        from pxr import UsdGeom, Usd, UsdPhysics
        from spark_agent import UnitreeG1WholeBodyIsaacAgent
        from spark_robot import UnitreeG1WholeBodyDynamic1Config
        self.cfg, self.args, self.app, self.torch = cfg, args, app, torch
        self.robot_cfg = UnitreeG1WholeBodyDynamic1Config()
        self.agent = UnitreeG1WholeBodyIsaacAgent(self.robot_cfg, num_envs=1,
            device=args.physics_device, dt=.005, control_decimation=4, scalar_api=False,
            enable_hand_control=False, render=not args.headless, render_on_step=False,
            defer_render=True, render_robot_collision_volumes=False, enable_keyboard_control=False,
            viewport_only=True, viewport_layout_updates=0,
            **LocomotionController.plant_config(args.policy))
        self.agent.attach_simulation_app(app)
        stage = omni.usd.get_context().get_stage()
        for path in ("/World/defaultGroundPlane", "/World/SparkEnvironment/GroundGrid"):
            if stage.GetPrimAtPath(path):
                stage.RemovePrim(path)
        root = UsdGeom.Xform.Define(stage, "/World/WalkEnvironment")
        stage.DefinePrim(str(root.GetPath()) + "/Visual").GetReferences().AddReference(str(cfg.cache / "visual.usdc"))
        from .appearance import MeshAppearance
        self.mesh_appearance = MeshAppearance(stage, args.mesh_color)
        collision = stage.DefinePrim(str(root.GetPath()) + "/Collision")
        collision.GetReferences().AddReference(str(cfg.cache / "collision.usdc"))
        if not stage.GetPrimAtPath("/World/WalkEnvironment/Collision/Surface").HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError("Prepared environment has no collision geometry")
        data = np.load(cfg.cache / "collision.npz")
        self.query = SurfaceQueries(data["vertices"], data["faces"])
        self.spawns = np.load(cfg.cache / "spawns.npy")
        self.floor = float(self.spawns[0, 2])
        self.policy = LocomotionController(args.policy, self.robot_cfg, args.physics_device)
        self.upper = self.agent.default_body_pos[:, 12:].clone()

    def reset(self, spawn):
        torch = self.torch
        x, y, floor, yaw = map(float, spawn)
        pose = torch.tensor([[x, y, floor + .793 + .02, 0., 0., math.sin(yaw / 2), math.cos(yaw / 2)]],
                            dtype=torch.float32, device=self.args.physics_device)
        self.agent.reset(root_pose_w=pose)
        self.policy.reset()
        self.upper = self.agent.default_body_pos[:, 12:].clone()
        self.floor = floor
        return self.agent.get_feedback()

    def step(self, command):
        feedback = self.agent.get_feedback()
        target, info = self.policy.infer(feedback, command, self.upper)
        self.agent.step(target, action_info=info)
        return self.agent.get_feedback()


class ReconstructionWalkPipeline:
    def __init__(self, cfg, args, app, worker=None):
        self.cfg, self.args, self.app = cfg, args, app
        self.environment = ReconstructionEnvironment(cfg, args, app)
        self.task = KeyboardWalkTask()
        self.renderer = None
        self.subscription = self.input = self.keyboard = None
        self.reset_count = 0
        self.steps = 0
        self.last_input_surface_focused = True
        if not args.headless:
            import carb.input
            import omni.appwindow
            self.input = carb.input.acquire_input_interface()
            self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self.key_event)
            from .rendering import DualViewRenderer
            self.renderer = DualViewRenderer(self.environment.agent, cfg, self.environment.query, worker,
                args.width, args.height, args.light_scale if args.light_scale is not None else cfg.data.get("light_scale", 2),
                mesh_appearance=self.environment.mesh_appearance, video_dir=args.video_dir,
                video_fps=args.video_fps, record=args.record)

    def key_event(self, event, *args):
        import carb.input
        # CHAR events carry text instead of a KeyboardInput enum. Only physical
        # press/release events should update held keys or trigger a reset.
        if event.type not in (carb.input.KeyboardEventType.KEY_PRESS,
                              carb.input.KeyboardEventType.KEY_RELEASE):
            return True
        key = getattr(event.input, "name", event.input)
        if not isinstance(key, str):
            return True
        key = "SPACE" if key == " " else key.rsplit(".", 1)[-1].strip().upper()
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.task.keys.discard(key)
        elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == "F9":
                if self.renderer is not None:
                    self.renderer.toggle_recording()
            elif key == "R":
                self.task.reset_requested = True
            elif key == "SPACE":
                self.task.clear_input()
                self.task.keys.add("SPACE")
            elif key in ("W", "A", "S", "D", "Q", "E"):
                self.task.keys.add(key)
        return True

    def _focus_input(self):
        if self.input is None:
            return
        import omni.appwindow
        if not omni.appwindow.get_default_app_window().is_focused():
            self.task.clear_input()
            return
        # Reconcile held keys against the native input state. This also clears
        # stale keys after the application loses focus and releases its inputs.
        import carb.input
        for name in tuple(self.task.keys):
            code = getattr(carb.input.KeyboardInput, name, None)
            if code is not None and self.input.get_keyboard_value(self.keyboard, code) == 0:
                self.task.keys.discard(name)

    def run(self):
        env = self.environment
        feedback = env.reset(env.spawns[0])
        root = feedback["root_pose_w"][0].detach().cpu().numpy()
        self.task.reset(root[:3])
        print(f"[walk] {self.args.policy}: spawn={env.spawns[0].tolist()}; WASD/QE, Space stop, R reset", flush=True)
        print("[walk] Scale:", json.loads((self.cfg.cache / "prepared.json").read_text())["calibration"], flush=True)
        started = time.monotonic()
        deadline = time.perf_counter()
        while self.app.is_running() and (self.args.steps < 0 or self.steps < self.args.steps):
            self._focus_input()
            if self.args.test_fall_at is not None and self.steps == self.args.test_fall_at:
                pose = env.agent.get_feedback()["root_pose_w"].clone()
                pose[:, 2] = env.floor + .25
                pose[:, 3:7] = env.torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)], device=pose.device)
                ids = env.torch.tensor([0], dtype=env.torch.long, device=pose.device)
                env.agent.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                env.agent.robot.write_root_velocity_to_sim_index(
                    root_velocity=env.torch.zeros((1, 6), device=pose.device), env_ids=ids)
                env.agent.scene.write_data_to_sim()
                env.agent.sim.forward()
                env.agent.scene.update(0.)
                print("[test] Injected a physical fall to validate nearby recovery", flush=True)
            command = self.task.compute_command()
            if self.args.scripted_command is not None and self.task.age >= 70:
                command = np.clip(self.args.scripted_command, -self.task.limits, self.task.limits)
            feedback = env.step(command)
            root = feedback["root_pose_w"][0].detach().cpu().numpy()
            if self.steps % 5 == 0 and np.isfinite(root).all():
                h = env.query.heights(root[None, :2], reference_z=env.floor)[0]
                if math.isfinite(h):
                    env.floor = float(h)
            reason = self.task.observe(root, env.floor)
            if reason is not None:
                if self.args.max_resets >= 0 and self.reset_count >= self.args.max_resets:
                    raise RuntimeError(f"Reset limit reached after {reason}; inspect contacts/spawn/policy")
                spawn = self.task.choose_spawn(env.spawns, root[:3])
                feedback = env.reset(spawn)
                self.task.reset(feedback["root_pose_w"][0, :3].detach().cpu().numpy())
                self.reset_count += 1
                print(f"[walk] Reset {self.reset_count}: {reason}; nearby spawn={spawn.tolist()}", flush=True)
                if self.renderer:
                    self.renderer.invalidate()
                root = feedback["root_pose_w"][0].detach().cpu().numpy()
            if self.renderer:
                self.renderer.publish_ready()
                if self.steps % self.args.render_every == 0:
                    self.renderer.capture(self.steps, self.args.policy, self.steps * .02, root)
            self.steps += 1
            if self.args.real_time:
                deadline += .02
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -.5:
                    deadline = time.perf_counter()
            if self.steps % 250 == 0:
                print(f"[walk] steps={self.steps} resets={self.reset_count} root={root[:3].round(3).tolist()} "
                      f"effective_hz={self.steps/max(time.monotonic()-started,1e-6):.1f}", flush=True)
        result = {"policy": self.args.policy, "steps": self.steps, "resets": self.reset_count,
                  "root_xyzw": root.tolist(), "wall_seconds": time.monotonic() - started,
                  "visual_state_id": self.renderer.last_sequence if self.renderer else None}
        (self.cfg.cache / f"run-{self.args.policy}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("[walk] Finished:", json.dumps(result), flush=True)

    def close(self):
        if self.subscription is not None:
            self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        if self.renderer:
            self.renderer.close()
        if getattr(self, "environment", None) is not None:
            self.environment.agent.close_viewer()
            self.environment.agent.close()
