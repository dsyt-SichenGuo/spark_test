"""Unified static-environment G1 walking runtime with configurable episodes.

Uses the original 50 Hz controller, WBT/Sport policy transitions, and
independent resets. Static assets contain no people or replay animation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import traceback


REPO_ROOT = Path(__file__).resolve().parents[3]


class EnvironmentTensorTask:
    """Batched pose tracking, with the original command and reset criteria."""

    def __init__(self, torch, *, env_origins, args):
        self.torch = torch
        self.device = env_origins.device
        self.num_envs = int(env_origins.shape[0])
        self.env_origins = env_origins
        self.control_dt = 0.02
        self.max_episode_steps = int(args.episode_steps)
        self.stall_window_steps = int(args.stall_window_steps)
        self.fall_height = float(args.fall_height)
        self.floor_heights = torch.zeros(self.num_envs, device=self.device)
        self.fall_up_z = math.cos(math.radians(args.fall_tilt_degrees))
        self.base_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_tolerance = args.goal_tolerance
        self.goal_yaw_tolerance = math.radians(args.goal_yaw_tolerance_degrees)
        self.linear_limit = torch.tensor(
            (args.max_forward_speed, args.max_lateral_speed), device=self.device
        )
        self.max_yaw_rate = args.max_yaw_rate
        self.velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_steps = torch.zeros_like(self.episode_steps)
        self.xy_history = torch.zeros(self.stall_window_steps, self.num_envs, 2, device=self.device)

    def set_goals(self, env_ids, local_goal_xy, goal_yaw):
        self.base_goals[env_ids, :2] = self.env_origins[env_ids, :2] + local_goal_xy
        self.base_goals[env_ids, 2] = goal_yaw

    def reset(self, env_ids, feedback):
        self.episode_steps[env_ids] = 0
        self.motion_steps[env_ids] = 0
        self.velocity_command[env_ids] = 0.0
        self.xy_history[:, env_ids] = feedback["root_pose_w"][env_ids, :2].unsqueeze(0)

    def target_heading(self, pose):
        """Face the actual target; at coincident XY use the episode approach heading."""
        torch = self.torch
        offset = self.base_goals[:, :2] - pose[:, :2]
        return torch.where(torch.linalg.vector_norm(offset, dim=1) > 1e-4,
                           torch.atan2(offset[:, 1], offset[:, 0]), self.base_goals[:, 2])

    def compute_command(self, feedback):
        torch = self.torch
        pose = feedback["root_pose_w"]
        qx, qy, qz, qw = pose[:, 3:7].unbind(dim=1)
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
        error_w = self.base_goals[:, :2] - pose[:, :2]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        error_b = torch.stack((cosine * error_w[:, 0] + sine * error_w[:, 1],
                               -sine * error_w[:, 0] + cosine * error_w[:, 1]), dim=1)
        velocity_w = feedback["root_velocity_w"][:, :2]
        velocity_b = torch.stack((cosine * velocity_w[:, 0] + sine * velocity_w[:, 1],
                                  -sine * velocity_w[:, 0] + cosine * velocity_w[:, 1]), dim=1)
        desired_xy = 0.8 * error_b - 0.4 * velocity_b
        desired_xy = torch.maximum(torch.minimum(desired_xy, self.linear_limit), -self.linear_limit)
        heading = self.target_heading(pose)
        yaw_error = torch.atan2(torch.sin(heading - yaw), torch.cos(heading - yaw))
        desired_yaw = torch.clamp(
            0.8 * yaw_error - 0.25 * feedback["root_angular_velocity_b"][:, 2],
            -self.max_yaw_rate, self.max_yaw_rate,
        )
        self.velocity_command[:, :2] += torch.clamp(
            desired_xy - self.velocity_command[:, :2], -0.006, 0.006
        )
        self.velocity_command[:, 2] += torch.clamp(
            desired_yaw - self.velocity_command[:, 2], -0.015, 0.015
        )
        return self.velocity_command

    def update_after_step(self, feedback):
        torch = self.torch
        self.episode_steps.add_(1)
        self.motion_steps.add_(1)
        pose = feedback["root_pose_w"]
        rows = torch.arange(self.num_envs, device=self.device)
        slots = (self.episode_steps - 1).remainder(self.stall_window_steps)
        previous_xy = self.xy_history[slots, rows].clone()
        self.xy_history[slots, rows] = pose[:, :2]
        up_z = 1 - 2 * (pose[:, 3].square() + pose[:, 4].square())
        fallen = (pose[:, 2] - self.floor_heights < self.fall_height) | (up_z < self.fall_up_z)
        goal_distance = torch.linalg.vector_norm(self.base_goals[:, :2] - pose[:, :2], dim=1)
        stalled = ((self.episode_steps > 250)
                   & (self.episode_steps >= self.stall_window_steps)
                   & (goal_distance > self.goal_tolerance)
                   & (torch.linalg.vector_norm(pose[:, :2] - previous_xy, dim=1) < 0.035))
        qx, qy, qz, qw = pose[:, 3:7].unbind(dim=1)
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy.square() + qz.square()))
        heading = self.target_heading(pose)
        yaw_error = torch.atan2(torch.sin(heading - yaw), torch.cos(heading - yaw))
        completed = ((self.motion_steps >= 50)
                     & (goal_distance <= self.goal_tolerance)
                     & (torch.abs(yaw_error) <= self.goal_yaw_tolerance))
        timed_out = self.episode_steps >= self.max_episode_steps
        stalled &= ~fallen
        completed &= ~fallen & ~stalled
        timed_out &= ~fallen & ~stalled & ~completed
        return {"fallen": fallen, "blocked_or_stalled": stalled,
                "completed": completed, "timeout": timed_out}


class GroundSurfaceSampler:
    """Find supported floor patches in an uneven reconstructed triangle mesh.

    Uses upward-facing triangles near the known floor level, rejecting holes,
    steep surfaces, and discontinuities under a 40 cm wide support footprint.
    """

    def __init__(self, triangles, reference_floor_z):
        import numpy as np
        self.np = np
        a = triangles[:, 0]
        u, v = triangles[:, 1] - a, triangles[:, 2] - a
        self.camera_a, self.camera_u, self.camera_v = a, u, v
        self.bounds_min = triangles.reshape(-1, 3).min(axis=0)
        self.bounds_max = triangles.reshape(-1, 3).max(axis=0)
        normal = np.cross(u, v)
        determinant = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        keep = ((normal[:, 2] / np.maximum(np.linalg.norm(normal, axis=1), 1e-9)
                 >= math.cos(math.radians(35))) & (np.abs(determinant) > 1e-8))
        self.a, self.u, self.v, self.det = a[keep], u[keep], v[keep], determinant[keep]
        self.reference_floor_z = reference_floor_z

    @classmethod
    def from_usd(cls, usd_path, mesh_path, world_z_offset, reference_floor_z):
        import numpy as np
        from pxr import Usd, UsdGeom, Gf
        stage = Usd.Stage.Open(str(usd_path))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(mesh_path))
        if not mesh:
            raise ValueError(f"terrain mesh not found: {mesh_path}")
        points = np.asarray(mesh.GetPointsAttr().Get())
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
        points = np.asarray([matrix.Transform(Gf.Vec3d(*map(float, p))) for p in points])
        points[:, 2] += world_z_offset
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        triangles = []
        offset = 0
        for count in counts:
            polygon = indices[offset:offset + count]
            triangles.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, count - 1))
            offset += count
        return cls(points[np.asarray(triangles)], reference_floor_z)

    def _height(self, x, y):
        np = self.np
        w = np.array([x, y]) - self.a[:, :2]
        b = (w[:, 0] * self.v[:, 1] - w[:, 1] * self.v[:, 0]) / self.det
        c = (self.u[:, 0] * w[:, 1] - self.u[:, 1] * w[:, 0]) / self.det
        z = self.a[:, 2] + b * self.u[:, 2] + c * self.v[:, 2]
        inside = ((b >= -1e-6) & (c >= -1e-6) & (b + c <= 1 + 1e-6)
                  & (np.abs(z - self.reference_floor_z) <= 0.75))
        return float(z[inside].max()) if np.any(inside) else float("nan")

    def __call__(self, xy):
        heights = []
        for x, y in xy:
            footprint = [self._height(float(x) + dx, float(y) + dy)
                         for dx, dy in ((0, 0), (.2, 0), (-.2, 0), (0, .2), (0, -.2))]
            heights.append(footprint[0] if all(math.isfinite(h) for h in footprint)
                           and max(footprint) - min(footprint) <= .2 else float("nan"))
        return self.np.asarray(heights)

    def camera_segment_fraction(self, start, end, clearance=.18):
        """Keep a camera segment inside the mesh bounds and in front of walls/roof."""
        np = self.np
        start, end = np.asarray(start), np.asarray(end)
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-8:
            return 1.
        p = np.cross(direction, self.camera_v)
        det = np.einsum('ij,ij->i', self.camera_u, p)
        valid = np.abs(det) > 1e-9
        inverse = np.zeros_like(det)
        inverse[valid] = 1. / det[valid]
        offset = start - self.camera_a
        u = np.einsum('ij,ij->i', offset, p) * inverse
        q = np.cross(offset, self.camera_u)
        v = (q @ direction) * inverse
        t = np.einsum('ij,ij->i', self.camera_v, q) * inverse
        hit = valid & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1 + 1e-6) & (t > 1e-5) & (t <= 1.)
        fraction = max(0., float(t[hit].min()) - clearance / length) if np.any(hit) else 1.
        for axis in (0, 1):
            if direction[axis] > 1e-8:
                fraction = min(fraction, (self.bounds_max[axis] - clearance - start[axis]) / direction[axis])
            elif direction[axis] < -1e-8:
                fraction = min(fraction, (self.bounds_min[axis] + clearance - start[axis]) / direction[axis])
        return max(0., min(1., fraction))


def sample_episode(torch, *, args, env_origins, xy_min, xy_max, spawn_base_z, ground_sampler=None):
    """Sample local poses, then translate them into each cloned world's frame."""
    count = int(env_origins.shape[0])
    device = env_origins.device
    low = torch.as_tensor(xy_min, dtype=env_origins.dtype, device=device)
    high = torch.as_tensor(xy_max, dtype=env_origins.dtype, device=device)
    def random_xy(n):
        return low + torch.rand(n, 2, device=device) * (high - low)

    robot_xy = (torch.tensor(args.robot_position, device=device).expand(count, -1).clone()
                if args.robot_position_mode == "manual" else random_xy(count))
    goal_xy = (torch.tensor(args.goal_position, device=device).expand(count, -1).clone()
               if args.goal_position_mode == "manual" else random_xy(count))
    for _ in range(256):
        close = torch.linalg.vector_norm(goal_xy - robot_xy, dim=1) < args.min_goal_distance
        invalid_start = torch.zeros(count, dtype=torch.bool, device=device)
        invalid_goal = torch.zeros_like(invalid_start)
        if ground_sampler is not None:
            start_floor = torch.as_tensor(ground_sampler(robot_xy.detach().cpu().numpy()), device=device)
            goal_floor = torch.as_tensor(ground_sampler(goal_xy.detach().cpu().numpy()), device=device)
            invalid_start = ~torch.isfinite(start_floor)
            invalid_goal = ~torch.isfinite(goal_floor)
            if args.robot_position_mode == "manual" and bool(invalid_start.any().item()):
                raise ValueError("fixed robot position has no supported floor; choose another --robot-position")
            if args.goal_position_mode == "manual" and bool(invalid_goal.any().item()):
                raise ValueError("fixed goal position has no supported floor; choose another --goal-position")
        if not bool((close | invalid_start | invalid_goal).any().item()):
            break
        if args.robot_position_mode == "random":
            resample = close | invalid_start
            robot_xy[resample] = random_xy(int(resample.sum().item()))
        if args.goal_position_mode == "random":
            resample = close | invalid_goal
            goal_xy[resample] = random_xy(int(resample.sum().item()))
    else:
        raise RuntimeError("cannot sample supported separated positions; adjust sampling bounds or goal distance")
    robot_yaw = (torch.full((count,), math.radians(args.robot_yaw_degrees), device=device)
                 if args.robot_yaw_mode == "manual"
                 else 2 * math.pi * torch.rand(count, device=device) - math.pi)
    offset = goal_xy - robot_xy
    # At arrival the facing direction is along this episode's start-to-goal line.
    goal_yaw = torch.atan2(offset[:, 1], offset[:, 0])
    spawn = torch.zeros(count, 7, device=device)
    spawn[:, :2] = env_origins[:, :2] + robot_xy
    spawn[:, 2] = env_origins[:, 2] + spawn_base_z
    if ground_sampler is not None and args.spawn_height is None:
        spawn[:, 2] = env_origins[:, 2] + start_floor.to(spawn.dtype) + 0.793
    spawn[:, 5] = torch.sin(robot_yaw / 2)
    spawn[:, 6] = torch.cos(robot_yaw / 2)
    return spawn, goal_xy, goal_yaw


def episode_camera_view(spawn_pose, goal_xy, *, distance, height, lookahead, segment_fraction=None):
    """Place camera XY on the start-to-goal line, behind the starting robot."""
    x, y, z = (float(v) for v in spawn_pose[:3])
    dx, dy = float(goal_xy[0]) - x, float(goal_xy[1]) - y
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        raise ValueError("camera requires distinct start and goal positions")
    dx, dy = dx / length, dy / length
    if segment_fraction is not None:
        # Prefer the requested angle; lower the camera when a ceiling blocks it.
        # Every candidate remains behind the start on the start-to-goal XY line.
        anchor = [x, y, z + .15]
        best = None
        for candidate_height in (height, height * .65, height * .4, height * .2):
            requested = [x - distance * dx, y - distance * dy, z + candidate_height]
            fraction = segment_fraction(anchor, requested)
            eye = [anchor[i] + fraction * (requested[i] - anchor[i]) for i in range(3)]
            score = distance * fraction
            if best is None or score > best[0] + 1e-6:
                best = (score, eye)
            if fraction >= 1. - 1e-6:
                break
        actual_distance, eye = best
        forward = min(lookahead, length, actual_distance * .15)
        return eye, [x + forward * dx, y + forward * dy, z]
    forward = min(lookahead, length)
    return ([x - distance * dx, y - distance * dy, z + height],
            [x + forward * dx, y + forward * dy, z + 0.3])


class StableFollowCamera:
    """Keep the initialized orbit radius/height, ignoring body roll, pitch and bob."""

    def __init__(self, pose, eye, target, smoothing):
        x, y, z = map(float, pose[:3])
        self.distance = math.hypot(eye[0] - x, eye[1] - y)
        self.heading = math.atan2(y - eye[1], x - eye[0])
        self.eye_height = float(eye[2]) - z
        self.target_height = float(target[2]) - z
        self.lookahead = math.hypot(target[0] - x, target[1] - y)
        self.initial_z = self.anchor_z = z
        self.smoothing = smoothing
        self.safe_fraction = 1.

    def update(self, pose, dt, ground_delta=0.):
        x, y = map(float, pose[:2])
        qx, qy, qz, qw = map(float, pose[3:7])
        gain = -math.expm1(-max(0., dt) / self.smoothing)
        # Retain heading during tumbles rather than flipping the camera with the body.
        up = 1. - 2. * (qx * qx + qy * qy)
        if up > .25 and all(math.isfinite(q) for q in (qx, qy, qz, qw)):
            yaw = math.atan2(2. * (qw * qz + qx * qy), 1. - 2. * (qy * qy + qz * qz))
            delta = math.atan2(math.sin(yaw - self.heading), math.cos(yaw - self.heading))
            self.heading += gain * delta
        if math.isfinite(ground_delta):
            self.anchor_z += gain * (self.initial_z + ground_delta - self.anchor_z)
        dx, dy = math.cos(self.heading), math.sin(self.heading)
        return ([x - self.distance * dx, y - self.distance * dy, self.anchor_z + self.eye_height],
                [x + self.lookahead * dx, y + self.lookahead * dy, self.anchor_z + self.target_height])

    def constrain(self, eye, anchor, *, bounds_min, bounds_max, dt,
                  segment_fraction=None, clearance=.18):
        """Retract immediately at obstacles; recover distance smoothly in free space."""
        low = [min(float(bounds_min[i]) + clearance, (bounds_min[i] + bounds_max[i]) / 2)
               for i in range(3)]
        high = [max(float(bounds_max[i]) - clearance, (bounds_min[i] + bounds_max[i]) / 2)
                for i in range(3)]
        # A fallen/out-of-bounds robot must not pull the camera out of the scene.
        anchor = [max(low[i], min(high[i], float(anchor[i]))) for i in range(3)]
        direction = [float(eye[i]) - anchor[i] for i in range(3)]
        fraction = 1.
        for i, delta in enumerate(direction):
            if delta > 1e-8:
                fraction = min(fraction, (high[i] - anchor[i]) / delta)
            elif delta < -1e-8:
                fraction = min(fraction, (low[i] - anchor[i]) / delta)
        if segment_fraction is not None:
            fraction = min(fraction, segment_fraction(anchor, eye, clearance=clearance))
        fraction = max(0., min(1., fraction))
        gain = -math.expm1(-max(0., dt) / self.smoothing)
        self.safe_fraction = min(fraction, self.safe_fraction + gain * (fraction - self.safe_fraction))
        # Clamp once more for floating-point roundoff at an exact boundary.
        return [max(low[i], min(high[i], anchor[i] + self.safe_fraction * direction[i])) for i in range(3)]


def create_interior_fill_light(stage, path, position, intensity):
    """Create a soft indoor fill that is not blocked by the enclosing roof."""
    from pxr import Gf, UsdLux
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(intensity)
    light.CreateRadiusAttr(.25)
    light.CreateNormalizeAttr(True)
    UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)
    translate = light.AddTranslateOp()
    translate.Set(Gf.Vec3d(*position))
    return translate


def create_marble_mesh_light(stage, path, position, mesh_path, *, shadows=True):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateIntensityAttr(6000.)
    light.CreateRadiusAttr(.35)
    translate = UsdGeom.Xformable(light).AddTranslateOp()
    translate.Set(Gf.Vec3d(*map(float, position)))
    link = UsdLux.LightAPI(light.GetPrim()).GetLightLinkCollectionAPI()
    link.CreateIncludeRootAttr(False)
    link.CreateExpansionRuleAttr(Usd.Tokens.expandPrims)
    link.CreateIncludesRel().SetTargets([Sdf.Path(mesh_path)])
    UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(shadows)
    return translate


def add_marble_mesh_corridor_lights(stage, env_origins):
    """Original Marble lamps, linked only to mesh excluded from the GS inset."""
    positions = [(-7.5, -14.0, 1.25), (-6.0, -11.5, 1.25),
                 (-4.3, -9.0, 1.25), (-2.5, -6.0, 1.25)]
    for env_id, origin in enumerate(env_origins):
        root = f"/World/StaticWalkTask/env_{env_id}"
        for index, position in enumerate(positions):
            # Original script positions are aligned world coordinates, not raw USD Z.
            create_marble_mesh_light(stage, f"{root}/Lighting/MarbleCorridor_{index}",
                                    [position[i] + float(origin[i]) for i in range(3)], f"{root}/Scene/Environment")


def add_static_environment_world(agent, args, asset):
    """Reference verified static subtrees, excluding humans and replay cameras."""
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

    manifest = json.loads((args.world_dir / "environment.json").read_text())
    usd_path = (args.world_dir / manifest["usd_path"]).resolve()
    source_stage = Usd.Stage.Open(str(usd_path))
    root_path = manifest["static_environment_prim"]
    source_root = source_stage.GetPrimAtPath(root_path)
    if not source_root:
        raise ValueError(f"Static environment root not found: {root_path} in {usd_path}")
    prims = list(Usd.PrimRange(source_root))
    if any(p.GetName() == "People" or p.IsA(UsdGeom.Camera) or p.IsA(UsdPhysics.Scene) for p in prims):
        raise ValueError("static environment must not contain People, cameras, or authored physics scenes")
    if any(attr.GetNumTimeSamples() for prim in prims
           for attr in prim.GetAttributes()):
        raise ValueError("environment must be static (no time-sampled attributes)")
    if not any(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims):
        raise ValueError("static environment has no collision geometry")
    stage = omni.usd.get_context().get_stage()
    # USD assets supply their own ground. SPARK's decorative grid is separate
    # from its ground plane and would otherwise float at z=0 above this terrain.
    for path in ("/World/defaultGroundPlane", "/World/SparkEnvironment/GroundGrid"):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
    for env_id, origin in enumerate(agent.scene.env_origins.detach().cpu().tolist()):
        scene = UsdGeom.Xform.Define(stage, f"/World/StaticWalkTask/env_{env_id}/Scene")
        scene.AddTranslateOp().Set(Gf.Vec3d(origin[0], origin[1], origin[2] + asset["world_z_offset"]))
        world = stage.DefinePrim(f"{scene.GetPath()}/Environment")
        world.GetReferences().AddReference(str(usd_path), root_path)
        if not world.GetChildren():
            raise RuntimeError("static environment reference did not resolve")
    dome = UsdLux.DomeLight.Define(stage, "/World/StaticWalkTask/DomeLight")
    dome.CreateIntensityAttr(asset["light_intensity"])
    print(f"[Environment tensor] asset={args.environment}, {args.num_envs} static scene(s); "
          "no people or replay animation", flush=True)


class EnvironmentTensorPipeline:
    """Pure tensor WBT/Sport walking in cloned static reconstructed worlds."""

    def __init__(self, args, simulation_app):
        import torch
        from spark_agent import UnitreeG1WholeBodyIsaacAgent
        from spark_agent.simulation.viewer_config import DEFAULT_VIEWER_CONFIG
        from spark_policy.control.whole_body.unitree_g1.sport import UnitreeG1SportPolicy
        from spark_policy.control.whole_body.unitree_g1.wbt import (
            UnitreeG1BatchedWBTPolicy, wbt_isaac_actuation_config,
        )
        from spark_robot import UnitreeG1WholeBodyDynamic1Config
        from scrpits.run_unitree_g1_usd_environment_walk_benchmark import ENVIRONMENT_ASSETS

        self.args, self.app, self.torch = args, simulation_app, torch
        self.asset = ENVIRONMENT_ASSETS[args.environment]
        self.spawn_base_z = args.spawn_height if args.spawn_height is not None else self.asset["spawn_base_z"]
        self.device = torch.device(args.device)
        self.compositor = None
        self.video_recorder = self.video_capture = None
        self.follow_camera = None
        self.render_enabled = not args.headless or args.record_video or ("gaussian_dir" in self.asset and args.gaussian_snapshot is not None)
        self.pipeline_step = 0
        self.reset_counts = dict.fromkeys(("fallen", "blocked_or_stalled", "completed", "timeout"), 0)
        self.agent = self.policy = None
        robot_cfg = UnitreeG1WholeBodyDynamic1Config()
        viewer_config = dict(DEFAULT_VIEWER_CONFIG)
        viewer_config.update(ground_style="plain", camera_lookat=(-4.8, -3.8, 0.5), camera_distance=10.0,
                             camera_azimuth=135.0, camera_elevation=-30.0)
        agent_kwargs = wbt_isaac_actuation_config() if args.policy == "wbtsafe" else {}
        self.agent = UnitreeG1WholeBodyIsaacAgent(
            robot_cfg, num_envs=args.num_envs, env_spacing=args.env_spacing,
            device=args.device, dt=0.005, control_decimation=4, scalar_api=False,
            enable_hand_control=False, render=self.render_enabled, render_on_step=False,
            defer_render=True, render_decimation=args.render_every, viewer_config=viewer_config,
            render_robot_collision_volumes=False,
            viewport_only=not (args.num_envs == 1 and self.render_enabled),
            viewport_layout_updates=0 if args.num_envs == 1 else 6, **agent_kwargs,
        )
        self.agent.attach_simulation_app(simulation_app)
        add_static_environment_world(self.agent, args, self.asset)
        self.ground_sampler = None
        if self.asset["terrain_mesh"] is not None:
            manifest = json.loads((args.world_dir / "environment.json").read_text())
            self.ground_sampler = GroundSurfaceSampler.from_usd(
                args.world_dir / manifest["usd_path"], self.asset["terrain_mesh"],
                self.asset["world_z_offset"], self.asset["spawn_base_z"] - 0.793,
            )
        self.camera_light = None
        self.mesh_camera_light = None
        if args.environment in ("marble", "marble-full"):
            import omni.usd
            import carb
            stage = omni.usd.get_context().get_stage()
            settings = carb.settings.get_settings()
            settings.set("/rtx/useViewLightingMode", False)
            settings.set("/rtx/sceneDb/ambientLightIntensity", args.interior_ambient_intensity)
            low, high = self.ground_sampler.bounds_min, self.ground_sampler.bounds_max
            for env_id, origin in enumerate(self.agent.scene.env_origins.detach().cpu().tolist()):
                for ix in range(4):
                    for iy in range(5):
                        x = float(low[0] + (ix + .5) * (high[0] - low[0]) / 4)
                        y = float(low[1] + (iy + .5) * (high[1] - low[1]) / 5)
                        floor = self.ground_sampler._height(x, y)
                        if not math.isfinite(floor):
                            floor = self.asset["spawn_base_z"] - .793
                        create_interior_fill_light(
                            stage, f"/World/StaticWalkTask/env_{env_id}/Lighting/Fill_{ix}_{iy}",
                            [x + origin[0], y + origin[1], floor + origin[2] + 1.3],
                            args.interior_light_intensity,
                        )
                        create_marble_mesh_light(
                            stage, f"/World/StaticWalkTask/env_{env_id}/Lighting/MarbleMeshFill_{ix}_{iy}",
                            [x + origin[0], y + origin[1], floor + origin[2] + 1.3],
                            f"/World/StaticWalkTask/env_{env_id}/Scene/Environment", shadows=False,
                        )
            self.camera_light = create_interior_fill_light(
                stage, "/World/StaticWalkTask/CameraFill", [0., 0., 1.], args.camera_light_intensity,
            )
            add_marble_mesh_corridor_lights(stage, self.agent.scene.env_origins.detach().cpu().tolist())
            self.mesh_camera_light = create_marble_mesh_light(
                stage, "/World/StaticWalkTask/MeshCameraFill", [0., 0., 1.],
                "/World/StaticWalkTask/env_0/Scene/Environment", shadows=False,
            )
        self.xy_min = torch.tensor(args.xy_min, device=self.device)
        self.xy_max = torch.tensor(args.xy_max, device=self.device)
        self.spawn_poses = torch.zeros(args.num_envs, 7, device=self.device)
        self.task = EnvironmentTensorTask(torch, env_origins=self.agent.scene.env_origins, args=args)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        all_envs = torch.arange(args.num_envs, device=self.device)
        self._sample_episode(all_envs)
        self.agent.reset(env_ids=all_envs, root_pose_w=self.spawn_poses)
        self.task.reset(all_envs, self.agent.get_feedback())
        self._update_goal_markers()
        if "gaussian_dir" in self.asset and self.render_enabled:
            from tools.marble_gaussian_compositor import MarbleGaussianCompositor
            self.compositor = MarbleGaussianCompositor(args, self.asset, self.device)
        self._refresh_viewer_camera(0)
        self.target = self.agent.default_body_pos.clone()
        self.upper_body_target = self.agent.default_body_pos[:, 12:].clone()
        policy_class = UnitreeG1BatchedWBTPolicy if args.policy == "wbtsafe" else UnitreeG1SportPolicy
        self.policy = policy_class(robot_cfg, num_envs=args.num_envs, device=args.device)
        self.policy.reset(env_ids=all_envs)
        if args.record_video:
            from tools.environment_walk_video import WalkVideoRecorder, NativeWalkVideoCapture
            self.video_recorder = WalkVideoRecorder(args)
            if self.compositor is None:
                self.video_capture = NativeWalkVideoCapture()

    def _sample_episode(self, env_ids):
        torch = self.torch
        if not env_ids.numel():
            return
        spawn, goal_xy, goal_yaw = sample_episode(
            torch, args=self.args, env_origins=self.agent.scene.env_origins[env_ids],
            xy_min=self.xy_min, xy_max=self.xy_max, spawn_base_z=self.spawn_base_z,
            ground_sampler=self.ground_sampler,
        )
        self.spawn_poses[env_ids] = spawn
        if self.ground_sampler is not None:
            local_xy = spawn[:, :2] - self.agent.scene.env_origins[env_ids, :2]
            floor = torch.as_tensor(self.ground_sampler(local_xy.detach().cpu().numpy()),
                                    device=self.device, dtype=self.task.floor_heights.dtype)
            self.task.floor_heights[env_ids] = self.agent.scene.env_origins[env_ids, 2] + floor
        else:
            self.task.floor_heights[env_ids] = self.agent.scene.env_origins[env_ids, 2]
        self.task.set_goals(env_ids, goal_xy, goal_yaw)
        print(f"[Environment tensor] initialized env_ids={env_ids.detach().cpu().tolist()}, "
              f"modes=({self.args.robot_position_mode}, {self.args.robot_yaw_mode}, "
              f"{self.args.goal_position_mode}), local_start_xy="
              f"{(spawn[:, :2] - self.agent.scene.env_origins[env_ids, :2]).detach().cpu().tolist()}, "
              f"local_goal_xy={goal_xy.detach().cpu().tolist()}, "
              f"spawn_z={spawn[:, 2].detach().cpu().tolist()}, "
              f"goal_yaw_degrees={torch.rad2deg(goal_yaw).detach().cpu().tolist()}", flush=True)

    def _update_goal_markers(self):
        markers = self.torch.zeros(self.args.num_envs, 3, device=self.device)
        markers[:, :2] = self.task.base_goals[:, :2] - self.agent.scene.env_origins[:, :2]
        markers[:, 2] = self.spawn_base_z
        if self.ground_sampler is not None:
            heights = self.ground_sampler(markers[:, :2].detach().cpu().numpy())
            markers[:, 2] = self.torch.as_tensor(heights, device=self.device) + 0.05
        self.agent.set_visual_goals(markers.detach().cpu().numpy(), base_radius=0.15)

    def _refresh_viewer_camera(self, env_id):
        if self.render_enabled:
            pose = self.spawn_poses[env_id].detach().cpu()
            goal = self.task.base_goals[env_id, :2].detach().cpu()
            origin = self.agent.scene.env_origins[env_id].detach().cpu()
            local_pose = pose.clone()
            local_pose[:3] -= origin
            eye, target = episode_camera_view(
                local_pose, goal - origin[:2],
                distance=self.args.camera_distance, height=self.args.camera_height,
                lookahead=self.args.camera_lookahead,
                segment_fraction=(self.ground_sampler.camera_segment_fraction
                                  if self.ground_sampler is not None else None),
            )
            eye = [eye[i] + float(origin[i]) for i in range(3)]
            target = [target[i] + float(origin[i]) for i in range(3)]
            self.camera_env_id = env_id
            self.camera_origin = origin.tolist()
            self.camera_last_step = self.pipeline_step
            self.camera_initial_floor = (self.ground_sampler._height(float(local_pose[0]), float(local_pose[1]))
                                         if self.ground_sampler is not None else 0.)
            self.follow_camera = (StableFollowCamera(pose, eye, target, self.args.camera_follow_smoothing)
                                  if self.args.camera_mode == "follow" else None)
            if hasattr(self, "viewer_camera"):
                self.viewer_camera.GetFocalLengthAttr().Set(18.)
                self.viewer_camera.GetHorizontalApertureAttr().Set(20.955)
            self._set_camera_view(eye, target)
            print(f"[Environment tensor] camera env_id={env_id}, mode={self.args.camera_mode}, "
                  f"eye={eye}, target={target}", flush=True)

    def _set_camera_view(self, eye, target):
        if not self.args.headless or self.args.record_video:
            import omni.usd
            from pxr import Gf, UsdGeom
            viewport = None
            if not self.args.headless:
                from omni.kit.viewport.utility import get_active_viewport
                viewport = get_active_viewport()
                if viewport is None:
                    raise RuntimeError("No active Isaac viewport for the task camera")
            if not hasattr(self, "viewer_camera"):
                stage = omni.usd.get_context().get_stage()
                self.viewer_camera = UsdGeom.Camera.Define(stage, "/World/StaticWalkTask/TaskCamera")
                self.viewer_camera.CreateClippingRangeAttr((.01, 1000.))
                self.viewer_camera.CreateFocalLengthAttr(18.)
                self.viewer_camera.CreateHorizontalApertureAttr(20.955)
                self.viewer_camera_transform = UsdGeom.Xformable(self.viewer_camera).MakeMatrixXform()
            camera = self.viewer_camera
            path = str(camera.GetPath())
            if viewport is not None and str(viewport.camera_path) != path:
                viewport.camera_path = path
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0., 0., 1.))
            self.viewer_camera_transform.Set(view.GetInverse())
            camera.CreateFocusDistanceAttr(math.dist(eye, target))
        if self.compositor is not None:
            self.compositor.set_view(eye, target, self.camera_origin)
        if self.camera_light is not None:
            from pxr import Gf
            self.camera_light.Set(Gf.Vec3d(*eye))
        if self.mesh_camera_light is not None:
            from pxr import Gf, Sdf, UsdLux
            self.mesh_camera_light.Set(Gf.Vec3d(*eye))
            light = self.mesh_camera_light.GetAttr().GetPrim()
            includes = UsdLux.LightAPI(light).GetLightLinkCollectionAPI().GetIncludesRel()
            targets = [Sdf.Path(f"/World/StaticWalkTask/env_{self.camera_env_id}/Scene/Environment")]
            if includes.GetTargets() != targets:
                includes.SetTargets(targets)

    def _update_follow_camera(self):
        if self.follow_camera is None:
            return
        pose = self.agent.get_feedback()["root_pose_w"][self.camera_env_id].detach().cpu().tolist()
        if not all(math.isfinite(v) for v in pose[:3]):
            return
        dt = (self.pipeline_step - self.camera_last_step) * .02
        self.camera_last_step = self.pipeline_step
        ground_delta = 0.
        if self.ground_sampler is not None:
            floor = self.ground_sampler._height(pose[0] - self.camera_origin[0], pose[1] - self.camera_origin[1])
            ground_delta = floor - self.camera_initial_floor
        eye, target = self.follow_camera.update(pose, dt, ground_delta)
        origin = self.camera_origin
        eye_local = [eye[i] - origin[i] for i in range(3)]
        anchor = [pose[0] - origin[0], pose[1] - origin[1], self.follow_camera.anchor_z - origin[2] + .15]
        eye_local = self.follow_camera.constrain(
            eye_local, anchor, bounds_min=self.args.camera_bounds_min, bounds_max=self.args.camera_bounds_max,
            dt=dt, clearance=self.args.camera_clearance,
            segment_fraction=(self.ground_sampler.camera_segment_fraction if self.ground_sampler is not None else None),
        )
        eye = [eye_local[i] + origin[i] for i in range(3)]
        self._set_camera_view(eye, target)

    def _infer_policy(self, feedback, command):
        torch = self.torch
        common = dict(body_joint_pos=feedback["body_joint_pos"],
                      body_joint_vel=feedback["body_joint_vel"],
                      root_quat_xyzw=feedback["root_pose_w"][:, 3:7],
                      root_angular_velocity=feedback["root_angular_velocity_b"],
                      upper_body_target=self.upper_body_target)
        warming = self.task.motion_steps < self.args.warmup_steps
        if self.args.policy == "sportsafe":
            sport_command = command.clone()
            sport_command[warming] = 0.0
            return self.policy.infer_tensor(**common, velocity_command=sport_command)
        common["stance"] = False
        loco_mask = ~warming
        blend = torch.clamp((self.task.motion_steps - self.args.warmup_steps + 1).float()
                            / max(1, self.args.transition_steps), 0.0, 1.0)
        if bool(torch.all(loco_mask).item()) and bool(torch.all(blend >= 1).item()):
            return self.policy.infer_tensor(**common, command=command, mode="loco")
        if bool(torch.all(~loco_mask).item()):
            return self.policy.infer_tensor(**common, command=None, mode="squat")
        squat, squat_info = self.policy.infer_tensor(**common, command=None, mode="squat")
        loco, loco_info = self.policy.infer_tensor(**common, command=command, mode="loco")
        target = torch.where(loco_mask[:, None],
                             (1 - blend[:, None]) * squat + blend[:, None] * loco, squat)
        info = dict(squat_info)
        for name in ("motor_kps", "motor_kds"):
            info[name] = torch.where(loco_mask[:, None],
                                     (1 - blend[:, None]) * squat_info[name]
                                     + blend[:, None] * loco_info[name], squat_info[name])
        return target, info

    def _reset_rows(self, masks):
        torch = self.torch
        reset_mask = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
        for reason, mask in masks.items():
            rows = torch.nonzero(mask & ~reset_mask).flatten()
            if rows.numel():
                self.reset_counts[reason] += int(rows.numel())
                reset_mask[rows] = True
                print(f"[Environment tensor] {reason}: env_ids={rows.detach().cpu().tolist()}", flush=True)
        env_ids = torch.nonzero(reset_mask).flatten()
        if not env_ids.numel():
            return
        self._sample_episode(env_ids)
        self.agent.reset(env_ids=env_ids, root_pose_w=self.spawn_poses[env_ids])
        self.policy.reset(env_ids=env_ids)
        self.task.reset(env_ids, self.agent.get_feedback())
        self._update_goal_markers()
        self._refresh_viewer_camera(int(env_ids[0].item()))
        self.target[env_ids] = self.agent.default_body_pos[env_ids]
        self.upper_body_target[env_ids] = self.agent.default_body_pos[env_ids, 12:]

    def step(self):
        feedback = self.agent.get_feedback()
        command = self.task.compute_command(feedback)
        self.target, policy_info = self._infer_policy(feedback, command)
        action_info = {"target_actuated_pos": self.target}
        action_info.update({name: policy_info[name] for name in ("motor_kps", "motor_kds")
                            if name in policy_info})
        self.agent.step(self.target, action_info=action_info)
        self._reset_rows(self.task.update_after_step(self.agent.get_feedback()))
        if self.render_enabled and self.pipeline_step % self.args.render_every == 0:
            self._render_frame()
        self.pipeline_step += 1

    def _render_frame(self, *, record=True):
        self._update_follow_camera()
        if self.compositor is None and self.video_capture is None:
            self.agent.render_frame()
            return
        # RTX may pump Kit; protect robot state while generating visual frames.
        state = self.agent._snapshot_camera_render_state()
        try:
            if self.compositor is not None:
                previous_frames = self.compositor.frames
                self.compositor.prepare()
            if self.video_capture is not None:
                self.video_capture.prepare(self.args.headless)
            self.agent.render_frame()
            if self.compositor is not None:
                self.compositor.update()
                if record and self.video_recorder is not None and self.compositor.frames > previous_frames:
                    self.video_recorder.append(self.compositor.last_native_image, self.compositor.last_image)
            elif record and self.video_recorder is not None:
                self.video_recorder.append(self.video_capture.frame())
        finally:
            self.agent._restore_camera_render_state(state)

    def run(self):
        print(f"[Environment tensor] running {self.args.num_envs} environment(s), "
              f"policy={self.args.policy}, scalar_api={self.agent.scalar_api}", flush=True)
        try:
            while self.args.steps < 0 or self.pipeline_step < self.args.steps:
                if getattr(self, "recording_stop", None) is not None and self.recording_stop.is_set():
                    print("[Video] stop requested; finalizing recordings", flush=True)
                    break
                if not self.app.is_running():
                    break
                started = time.perf_counter()
                self.step()
                if self.args.real_time:
                    remaining = 0.02 - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            print("[Environment tensor] interrupted", flush=True)
        if self.compositor is not None and self.args.gaussian_snapshot is not None:
            # Submit several frames so the final capture includes the latest reset.
            for _ in range(5):
                self._render_frame(record=False)
            self.compositor.save(self.args.gaussian_snapshot)
        feedback = self.agent.get_feedback()
        if not bool(self.torch.isfinite(feedback["root_pose_w"]).all().item()):
            raise RuntimeError("non-finite final robot pose")
        print(f"[Environment tensor] steps={self.pipeline_step}, resets={self.reset_counts}, "
              f"final_root_pose_w={feedback['root_pose_w'].detach().cpu().tolist()}", flush=True)

    def close(self):
        try:
            if getattr(self, "video_recorder", None) is not None:
                self.video_recorder.close()
        finally:
            if getattr(self, "video_capture", None) is not None:
                self.video_capture.close()
        if getattr(self, "compositor", None) is not None:
            self.compositor.close()
        if self.policy is not None and hasattr(self.policy, "close"):
            self.policy.close()
        if self.agent is not None:
            self.agent.close_viewer()
            self.agent.close()


def main():
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(REPO_ROOT))
    from scrpits.run_unitree_g1_usd_environment_walk_benchmark import (
        POLICY_RUNTIME, build_parser, validate_args,
    )
    parser = build_parser(for_runtime=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    validate_args(args, parser)
    args.policy = POLICY_RUNTIME[args.policy_config] if args.policy_config else (args.policy or "wbtsafe")
    if not args.headless:
        args.visualizer = "kit"
        args.max_visible_envs = args.num_envs
        args.hide_ui = False
        if not args.experience:
            args.experience = "isaacsim.exp.base.kit" if args.num_envs == 1 else "isaacsim.exp.base.python.kit"
    if args.record_video:
        args.enable_cameras = True
    if args.environment == "marble-full" and (not args.headless or args.gaussian_snapshot is not None or args.record_video):
        from tools.marble_gaussian_compositor import preflight_gaussian_renderer
        preflight_gaussian_renderer(args.device)
        args.enable_cameras = True
    launcher = AppLauncher(args)
    recording_stop = threading.Event()
    if args.record_video:
        # SimulationApp's SIGINT handler force-unloads Kit before Python finally
        # blocks can release encoders. Finish a complete paired frame instead.
        def request_recording_stop(signum, frame):
            recording_stop.set()
        signal.signal(signal.SIGINT, request_recording_stop)
        signal.signal(signal.SIGTERM, request_recording_stop)
    pipeline = EnvironmentTensorPipeline.__new__(EnvironmentTensorPipeline)
    pipeline.agent = pipeline.policy = None
    exit_code = 0
    try:
        pipeline.__init__(args, launcher.app)
        pipeline.recording_stop = recording_stop
        pipeline.run()
    except Exception:
        # Kit shutdown can terminate the interpreter; report failures first.
        exit_code = 1
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        try:
            pipeline.close()
        finally:
            launcher.app.close(wait_for_replicator=False, exit_code=exit_code)


if __name__ == "__main__":
    main()
