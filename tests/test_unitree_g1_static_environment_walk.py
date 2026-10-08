"""Episode contracts for the unified reconstructed-environment walk."""

import importlib.util
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load_module("environment_launcher", "scrpits/run_unitree_g1_usd_environment_walk_benchmark.py")
runtime = load_module("environment_runtime", "pipeline/spark_pipeline/autonomy/unitree_g1_environment_tensor_walk.py")


def configured_args(environment, mode):
    parser = launcher.build_parser()
    argv = ["--environment", environment, "--init-mode", mode]
    if mode == "fixed-start":
        argv += ["--robot-position", "-4", "-4", "--robot-yaw-degrees", "90"]
    elif mode == "fixed-goal":
        argv += ["--goal-position", "-4", "-4"]
    args = parser.parse_args(argv)
    launcher.validate_args(args, parser)
    return args


@pytest.mark.parametrize("environment", ["marble", "gpt-v4", "marble-full"])
@pytest.mark.parametrize("position_mode", ["random", "manual"])
@pytest.mark.parametrize("yaw_mode", ["random", "manual"])
@pytest.mark.parametrize("goal_mode", ["random", "manual"])
def test_independent_episode_modes(environment, position_mode, yaw_mode, goal_mode):
    parser = launcher.build_parser()
    argv = ["--environment", environment,
            "--robot-position-mode", position_mode, "--robot-yaw-mode", yaw_mode,
            "--goal-position-mode", goal_mode]
    if position_mode == "manual":
        argv += ["--robot-position", "-4", "-4"]
    if yaw_mode == "manual":
        argv += ["--robot-yaw-degrees", "90"]
    if goal_mode == "manual":
        argv += ["--goal-position", "2", "2"]
    args = parser.parse_args(argv)
    launcher.validate_args(args, parser)
    origins = torch.zeros(64, 3)
    origins[:, 0] = torch.arange(64) * 20
    torch.manual_seed(42)
    command = launcher.build_runtime_command(args)
    for name, expected in (("robot-position-mode", position_mode), ("robot-yaw-mode", yaw_mode),
                           ("goal-position-mode", goal_mode)):
        assert command[command.index("--" + name) + 1] == expected
    previous = None
    for _ in range(3):
        spawn, goal, heading = runtime.sample_episode(
            torch, args=args, env_origins=origins, xy_min=args.xy_min,
            xy_max=args.xy_max, spawn_base_z=launcher.ENVIRONMENT_ASSETS[environment]["spawn_base_z"],
        )
        start = spawn[:, :2] - origins[:, :2]
        offset = goal - start
        assert torch.all(torch.linalg.vector_norm(offset, dim=1) >= args.min_goal_distance)
        torch.testing.assert_close(heading, torch.atan2(offset[:, 1], offset[:, 0]), atol=1e-4, rtol=0)
        yaw = 2 * torch.atan2(spawn[:, 5], spawn[:, 6])
        if position_mode == "manual":
            torch.testing.assert_close(start, torch.full((64, 2), -4.0))
        else:
            assert torch.all(start >= torch.tensor(args.xy_min) - 1e-4)
            assert torch.all(start <= torch.tensor(args.xy_max) + 1e-4)
        if yaw_mode == "manual":
            torch.testing.assert_close(yaw, torch.full((64,), math.pi / 2))
        else:
            assert yaw.std() > .5
        if goal_mode == "manual":
            torch.testing.assert_close(goal, torch.full((64, 2), 2.0))
        else:
            assert torch.all(goal >= torch.tensor(args.xy_min))
            assert torch.all(goal <= torch.tensor(args.xy_max))
        if previous is not None:
            for mode, current, old in zip((position_mode, yaw_mode, goal_mode), (start, yaw, goal), previous):
                if mode == "random":
                    assert not torch.equal(current, old)
                else:
                    torch.testing.assert_close(current, old)
        previous = (start.clone(), yaw.clone(), goal.clone())


@pytest.mark.parametrize("environment", ["marble", "gpt-v4", "marble-full"])
def test_default_modes_are_random_and_runtime_command_preserves_modes(environment):
    parser = launcher.build_parser()
    args = parser.parse_args(["--environment", environment])
    launcher.validate_args(args, parser)
    command = launcher.build_runtime_command(args)
    for name in ("robot_position_mode", "robot_yaw_mode", "goal_position_mode"):
        assert getattr(args, name) == "random"
        assert command[command.index("--" + name.replace("_", "-")) + 1] == "random"


def test_manual_values_select_modes_and_both_fixed_positions_validate_distance():
    parser = launcher.build_parser()
    args = parser.parse_args(["--robot-position", "-4", "-4", "--goal-position", "2", "2"])
    launcher.validate_args(args, parser)
    assert (args.robot_position_mode, args.robot_yaw_mode, args.goal_position_mode) == ("manual", "random", "manual")


def test_arrival_faces_current_target_from_side_and_handles_coincident_position():
    args = configured_args("gpt-v4", "random")
    task = runtime.EnvironmentTensorTask(torch, env_origins=torch.zeros(3, 3), args=args)
    task.set_goals(torch.arange(3), torch.tensor([[4., 0.]] * 3), torch.zeros(3))
    # Same near-goal position: yaw=0 is wrong; yaw=-90 faces the target.
    pose = torch.tensor([[4., .1, .793, 0., 0., 0., 1.],
                         [4., .1, .793, 0., 0., -2**-.5, 2**-.5],
                         [4., 0., .793, 0., 0., 0., 1.]])
    task.motion_steps[:] = 50
    feedback = {"root_pose_w": pose, "root_velocity_w": torch.zeros(3, 6),
                "root_angular_velocity_b": torch.zeros(3, 3)}
    masks = task.update_after_step(feedback)
    assert masks["completed"].tolist() == [False, True, True]
    command = task.compute_command(feedback)
    assert command[0, 2] < 0
    assert abs(command[1, 2]) < 1e-6


def test_arrival_requires_heading_and_partial_reset_keeps_other_rows():
    args = configured_args("gpt-v4", "random")
    task = runtime.EnvironmentTensorTask(torch, env_origins=torch.zeros(2, 3), args=args)
    task.set_goals(torch.arange(2), torch.tensor([[4., 0.], [4., 0.]]), torch.zeros(2))
    pose = torch.tensor([[3.9, 0., .793, 0., 0., 0., 1.],
                         [3.9, 0., .793, 0., 0., 1., 0.]])
    task.reset(torch.arange(2), {"root_pose_w": pose})
    task.motion_steps[:] = 49
    task.episode_steps[:] = 49
    masks = task.update_after_step({"root_pose_w": pose})
    assert masks["completed"].tolist() == [True, False]
    task.velocity_command[:] = .1
    task.reset(torch.tensor([0]), {"root_pose_w": pose})
    assert task.episode_steps.tolist() == [0, 50]
    assert task.velocity_command[0].count_nonzero() == 0
    assert torch.all(task.velocity_command[1] == .1)


def test_turning_at_goal_is_not_stalled_and_fall_height_is_relative_to_floor():
    args = configured_args("marble", "random")
    task = runtime.EnvironmentTensorTask(torch, env_origins=torch.zeros(2, 3), args=args)
    task.set_goals(torch.arange(2), torch.tensor([[4., 0.], [4., 0.]]), torch.zeros(2))
    pose = torch.tensor([[3.9, 0., .013, 0., 0., 1., 0.],
                         [3.9, 0., .1, 0., 0., 0., 1.]])
    task.floor_heights[:] = torch.tensor([-.78, 0.])
    task.reset(torch.arange(2), {"root_pose_w": pose})
    task.episode_steps[:] = 300
    task.motion_steps[:] = 300
    masks = task.update_after_step({"root_pose_w": pose})
    assert masks["fallen"].tolist() == [False, True]
    assert masks["blocked_or_stalled"].tolist() == [False, False]
    assert masks["completed"].tolist() == [False, False]


def test_ground_sampler_rejects_holes_and_places_base_above_a_sloping_floor():
    import numpy as np
    # z = 0.1 * x - 0.5, covering a square floor and no surrounding surface.
    vertices = np.array([[-4., -4., -.9], [4., -4., -.1], [4., 4., -.1], [-4., 4., -.9]])
    ground = runtime.GroundSurfaceSampler(vertices[[[0, 1, 2], [0, 2, 3]]], -.5)
    assert np.allclose(ground([[0., 0.], [1., 1.]]), [-.5, -.4])
    assert np.isnan(ground([[5., 0.], [3.95, 0.]])).all()
    args = configured_args("marble", "random")
    args.min_goal_distance = 1.
    torch.manual_seed(12)
    spawn, goal, yaw = runtime.sample_episode(
        torch, args=args, env_origins=torch.zeros(10, 3), xy_min=[-5., -5.],
        xy_max=[5., 5.], spawn_base_z=.313, ground_sampler=ground,
    )
    torch.testing.assert_close(spawn[:, 2], .1 * spawn[:, 0] - .5 + .793)
    assert np.isfinite(ground(spawn[:, :2].numpy())).all()
    assert np.isfinite(ground(goal.numpy())).all()


def test_pipeline_partial_initialization_converts_ground_heights_to_tensor_dtype():
    import numpy as np
    from types import SimpleNamespace
    args = configured_args("marble", "random")
    args.min_goal_distance = 1.
    vertices = np.array([[-4., -4., -.9], [4., -4., -.1], [4., 4., -.1], [-4., 4., -.9]])
    pipeline = runtime.EnvironmentTensorPipeline.__new__(runtime.EnvironmentTensorPipeline)
    pipeline.args, pipeline.torch, pipeline.device = args, torch, torch.device("cpu")
    pipeline.xy_min, pipeline.xy_max = [-3., -3.], [3., 3.]
    pipeline.spawn_base_z = .313
    pipeline.ground_sampler = runtime.GroundSurfaceSampler(vertices[[[0, 1, 2], [0, 2, 3]]], -.5)
    origins = torch.tensor([[0., 0., 0.], [20., 0., 0.]])
    pipeline.agent = SimpleNamespace(scene=SimpleNamespace(env_origins=origins))
    pipeline.spawn_poses = torch.zeros(2, 7)
    pipeline.task = runtime.EnvironmentTensorTask(torch, env_origins=origins, args=args)
    pipeline._sample_episode(torch.tensor([1]))
    assert pipeline.spawn_poses[0].count_nonzero() == 0
    assert pipeline.task.floor_heights.dtype == torch.float32
    assert pipeline.task.floor_heights[0] == 0
    assert torch.isfinite(pipeline.task.floor_heights[1])
    torch.testing.assert_close(pipeline.spawn_poses[1, 2] - pipeline.task.floor_heights[1], torch.tensor(.793))


@pytest.mark.parametrize("start,goal", [([0., 0., .793], [3., 4.]),
                                       ([20., -10., .313], [17., -14.])])
def test_camera_is_behind_start_on_start_goal_line(start, goal):
    eye, target = runtime.episode_camera_view(start, goal, distance=3., height=2., lookahead=1.5)
    direction = torch.tensor(goal) - torch.tensor(start[:2])
    behind = torch.tensor(eye[:2]) - torch.tensor(start[:2])
    ahead = torch.tensor(target[:2]) - torch.tensor(start[:2])
    assert abs(float(direction[0] * behind[1] - direction[1] * behind[0])) < 1e-5
    assert float(direction @ behind) < 0
    assert float(direction @ ahead) > 0
    assert math.isclose(float(torch.linalg.vector_norm(behind)), 3., abs_tol=1e-5)
    assert math.isclose(eye[2], start[2] + 2.)


def test_indoor_camera_stays_below_ceiling_and_in_front_of_wall():
    import numpy as np
    # A ceiling at z=1.5 and a wall at x=-4, both double-sided for camera tests.
    ceiling = [[[-4., -4., 1.5], [4., -4., 1.5], [4., 4., 1.5]],
               [[-4., -4., 1.5], [4., 4., 1.5], [-4., 4., 1.5]]]
    wall = [[[-4., -4., -.5], [-4., 4., -.5], [-4., 4., 1.5]],
            [[-4., -4., -.5], [-4., 4., 1.5], [-4., -4., 1.5]]]
    ground = runtime.GroundSurfaceSampler(np.asarray(ceiling + wall), -.5)
    for start in ([0., 0., .293], [-3.5, 0., .293]):
        eye, target = runtime.episode_camera_view(
            start, [3., 0.], distance=3., height=2., lookahead=1.5,
            segment_fraction=ground.camera_segment_fraction,
        )
        assert -4. < eye[0] < start[0]
        assert eye[2] < 1.5
        assert eye[1] == 0.
        assert ground.camera_segment_fraction([*start[:2], start[2] + .15], eye) == 1.
        assert target[0] - start[0] <= (start[0] - eye[0]) * .25 + 1e-6


def test_indoor_fill_light_is_authored_with_shadows_disabled():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdLux
    stage = Usd.Stage.CreateInMemory()
    translate = runtime.create_interior_fill_light(stage, '/Fill', [1., 2., 1.3], 1600.)
    light = UsdLux.SphereLight(stage.GetPrimAtPath('/Fill'))
    assert light.GetIntensityAttr().Get() == 1600.
    assert UsdLux.ShadowAPI(light.GetPrim()).GetShadowEnableAttr().Get() is False
    assert tuple(translate.Get()) == (1., 2., 1.3)


def test_original_marble_corridor_lights_only_illuminate_each_clone_mesh():
    from pxr import Usd, UsdGeom, UsdLux
    stage = Usd.Stage.CreateInMemory()
    for env_id in range(2):
        UsdGeom.Mesh.Define(stage, f"/World/StaticWalkTask/env_{env_id}/Scene/Environment/Mesh")
        UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_id}/Robot/body")
    runtime.add_marble_mesh_corridor_lights(stage, [[0., 0., 0.], [20., 0., 0.]])
    for env_id in range(2):
        for index in range(4):
            light = UsdLux.SphereLight(stage.GetPrimAtPath(
                f"/World/StaticWalkTask/env_{env_id}/Lighting/MarbleCorridor_{index}"))
            assert light.GetIntensityAttr().Get() == 6000.
            assert light.GetRadiusAttr().Get() == pytest.approx(.35)
            assert light.GetNormalizeAttr().Get() is False  # Original lamp default.
            query = UsdLux.LightAPI(light.GetPrim()).GetLightLinkCollectionAPI().ComputeMembershipQuery()
            assert query.IsPathIncluded(f"/World/StaticWalkTask/env_{env_id}/Scene/Environment/Mesh")
            assert not query.IsPathIncluded(f"/World/envs/env_{env_id}/Robot/body")
            assert not query.IsPathIncluded(f"/World/StaticWalkTask/env_{1-env_id}/Scene/Environment/Mesh")
        matrix = UsdGeom.Xformable(stage.GetPrimAtPath(
            f"/World/StaticWalkTask/env_{env_id}/Lighting/MarbleCorridor_0")).ComputeLocalToWorldTransform(0)
        assert tuple(matrix.ExtractTranslation()) == pytest.approx((-7.5 + env_id * 20., -14., 1.25))


@pytest.mark.parametrize("environment", ["marble", "gpt-v4", "marble-full"])
def test_selected_assets_have_collisions_and_no_people_or_animation(environment):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics
    import json

    asset = launcher.ENVIRONMENT_ASSETS[environment]
    manifest = json.loads((asset["world_dir"] / "environment.json").read_text())
    stage = Usd.Stage.Open(str(asset["world_dir"] / manifest["usd_path"]))
    prims = list(stage.Traverse())
    assert any(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims)
    assert not any(p.GetName() == "People" or p.IsA(UsdGeom.Camera) or p.IsA(UsdPhysics.Scene) for p in prims)
    assert not any(a.GetNumTimeSamples() for p in prims for a in p.GetAttributes())


@pytest.mark.parametrize("argv", [
    ["--init-mode", "fixed-goal"],
    ["--init-mode", "random", "--robot-yaw-degrees", "0"],
    ["--init-mode", "fixed-start", "--goal-position", "0", "0"],
    ["--init-mode", "fixed-goal", "--goal-position", "100", "100"],
    ["--robot-position-mode", "manual"],
    ["--robot-yaw-mode", "manual"],
    ["--goal-position-mode", "manual"],
    ["--robot-position-mode", "random", "--robot-position", "0", "0"],
    ["--robot-yaw-mode", "random", "--robot-yaw-degrees", "0"],
    ["--goal-position-mode", "random", "--goal-position", "0", "0"],
    ["--robot-position", "0", "0", "--goal-position", "0", "0"],
    ["--init-mode", "fixed-start", "--robot-position-mode", "random"],
    ["--sampling-bounds", "0", "0", "0", "1"],
    ["--min-goal-distance", "1000"],
])
def test_invalid_initialization_fails_before_isaac_starts(argv):
    parser = launcher.build_parser()
    args = parser.parse_args(argv)
    with pytest.raises(SystemExit):
        launcher.validate_args(args, parser)


@pytest.mark.parametrize("unsupported", ["robot", "goal"])
def test_manual_unsupported_floor_fails_without_randomizing_anchor(unsupported):
    import numpy as np
    parser = launcher.build_parser()
    args = parser.parse_args(["--robot-position", "-4", "-4", "--goal-position", "2", "2"])
    launcher.validate_args(args, parser)
    def ground(xy):
        heights = np.zeros(len(xy))
        heights[xy[:, 0] == (-4 if unsupported == "robot" else 2)] = np.nan
        return heights
    with pytest.raises(ValueError, match=f"fixed {unsupported} position has no supported floor"):
        runtime.sample_episode(torch, args=args, env_origins=torch.zeros(1, 3),
                               xy_min=args.xy_min, xy_max=args.xy_max,
                               spawn_base_z=.793, ground_sampler=ground)


def test_gaussian_alignment_matches_every_original_collider_vertex():
    import json
    import numpy as np
    import trimesh
    from pxr import Usd
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    asset = launcher.ENVIRONMENT_ASSETS["marble-full"]
    alignment = json.loads(asset["gaussian_alignment"].read_text())
    rotation, shift, scale = module.gaussian_to_collider_transform(alignment)
    mesh = trimesh.load(asset["gaussian_dir"] / "collider_mesh.glb", force="scene").to_geometry()
    points = np.asarray(mesh.vertices) @ rotation.T * scale + shift
    stage = Usd.Stage.Open(str(asset["world_dir"] / "marble-1.1-static.usdc"))
    expected = np.asarray(stage.GetPrimAtPath("/World/MarbleCollider").GetAttribute("points").Get())
    np.testing.assert_allclose(points, expected, atol=1e-6, rtol=0)


def test_depth_composition_occludes_robot_behind_shelf_and_handles_empty_background():
    import numpy as np
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    gs = np.zeros((1, 4, 3), dtype=np.uint8)
    native = np.full_like(gs, 200)
    z = np.array([[2., 2., 2., 2.]])
    nz = np.array([[1., 3., np.inf, 3.]])
    alpha = np.array([[1., 1., 1., 0.]])
    image, stats = module.compose_depth(gs, z, alpha, native, nz)
    np.testing.assert_array_equal(image[0, :, 0], [200, 0, 0, 200])
    assert stats == {"native_pixels": 3, "visible_native_pixels": 2, "occluded_native_pixels": 1}


def test_gaussian_assets_decode_with_valid_unit_quaternions():
    import numpy as np
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    asset = launcher.ENVIRONMENT_ASSETS["marble-full"]
    xyz, quats, scales, rgb, alpha = module.load_spz(asset["gaussian_dir"] / "world_100k.spz")
    assert xyz.shape == (98304, 3)
    assert np.isfinite(xyz).all() and (scales > 0).all()
    np.testing.assert_allclose(np.linalg.norm(quats, axis=1), 1., atol=1e-6)
    assert ((rgb >= 0) & (rgb <= 1)).all()
    assert ((alpha >= 0) & (alpha <= 1)).all()


def test_full_appearance_does_not_change_seeded_task_sampling():
    parser = launcher.build_parser()
    results = []
    for environment in ("marble", "marble-full"):
        args = parser.parse_args(["--environment", environment])
        launcher.validate_args(args, parser)
        asset = launcher.ENVIRONMENT_ASSETS[environment]
        torch.manual_seed(19)
        results.append(runtime.sample_episode(torch, args=args, env_origins=torch.zeros(4, 3),
                                              xy_min=args.xy_min, xy_max=args.xy_max,
                                              spawn_base_z=asset["spawn_base_z"]))
    for original, full in zip(*results):
        torch.testing.assert_close(original, full, atol=0, rtol=0)


def test_static_gaussian_cache_refreshes_native_robot_and_invalidates_on_view_change():
    import numpy as np
    from types import SimpleNamespace
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    compositor = module.MarbleGaussianCompositor.__new__(module.MarbleGaussianCompositor)
    compositor.width = compositor.height = 2
    compositor.gui = False
    compositor.frames = compositor.gaussian_renders = 0
    compositor._cached_view = compositor._cached_gaussians = None
    compositor.pending_c2w = np.eye(4)
    lens = [18.]
    compositor.camera = SimpleNamespace(
        GetFocalLengthAttr=lambda: SimpleNamespace(Get=lambda: lens[0]),
        GetHorizontalApertureAttr=lambda: SimpleNamespace(Get=lambda: 20.955))
    native = np.ones((2, 2, 3), dtype=np.uint8)
    compositor.rgb = SimpleNamespace(get_data=lambda: native)
    compositor.depth = SimpleNamespace(get_data=lambda: np.ones((2, 2)))
    compositor.instances = SimpleNamespace(get_data=lambda: {
        "data": np.ones((2, 2), dtype=np.uint32),
        "info": {"idToLabels": {1: "/World/envs/env_0/Robot/body"}}})
    calls = []
    def render(camera):
        calls.append(camera.copy())
        return np.zeros_like(native), np.full((2, 2), 2.), np.ones((2, 2))
    compositor._render_gaussians = render
    compositor.update()
    native[:] = 200  # Robot moves while the static camera/scene stays fixed.
    compositor.update()
    assert len(calls) == 1
    np.testing.assert_array_equal(compositor.last_image, native)
    compositor.pending_c2w[0, 3] = 20.  # Episode reset or selected clone changes.
    compositor.update()
    lens[0] = 24.  # Camera zoom also changes the projected background/depth.
    compositor.update()
    assert len(calls) == 3 and compositor.frames == 4


def test_gs_inset_excludes_visible_mesh_but_keeps_robot_and_goal_depth():
    import numpy as np
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    original = np.ones((1, 4), dtype=np.float32)
    instances = {"data": np.array([[1, 2, 3, 4]], dtype=np.uint32), "info": {"idToLabels": {
        "1": "/World/StaticWalkTask/env_0/Scene/Environment/MarbleCollider",
        2: "/World/envs/env_0/Robot/body", 3: "/World/StaticWalkTask/Goals/Goal_0",
        4: "/World/StaticWalkTask/env_1/Scene/Environment/MarbleCollider"}}}
    depth, count = module.exclude_environment_depth(original, instances)
    np.testing.assert_array_equal(depth, [[np.inf, 1., 1., np.inf]])
    np.testing.assert_array_equal(original, np.ones((1, 4)))
    assert count == 2


@pytest.mark.parametrize("inherited,expected", [("10.1", "8.9"), ("", "8.9"),
                                                ("8.9", "8.9"), ("8.9+PTX", "8.9+PTX")])
def test_gaussian_architecture_repairs_invalid_shell_override(monkeypatch, inherited, expected):
    import os
    from types import SimpleNamespace
    module = load_module("gaussian_compositor", "tools/marble_gaussian_compositor.py")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", inherited)
    selected = []
    def capability(device):
        selected.append(device)
        return 8, 9
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=capability))
    module.configure_gaussian_architecture(fake_torch, "cuda:1")
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == expected
    assert selected == ["cuda:1"]


@pytest.mark.parametrize("environment", ["marble", "marble-full", "gpt-v4"])
def test_camera_modes_forward_to_runtime(environment):
    parser = launcher.build_parser()
    assert parser.parse_args([]).camera_mode == "fixed"
    args = parser.parse_args(["--environment", environment, "--camera-mode", "follow"])
    launcher.validate_args(args, parser)
    command = launcher.build_runtime_command(args)
    assert command[command.index("--camera-mode") + 1] == "follow"


def test_follow_camera_preserves_initial_view_distance_and_ignores_body_bob():
    pose = [1., 2., .8, 0., 0., 0., 1.]
    eye, target = runtime.episode_camera_view(pose, [7., 5.], distance=3., height=2., lookahead=1.5)
    camera = runtime.StableFollowCamera(pose, eye, target, .35)
    initial = camera.update(pose, 0.)
    assert initial[0] == pytest.approx(eye)
    assert initial[1] == pytest.approx(target)
    moved = [5., -3., .1, 0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]
    for _ in range(200):
        actual_eye, actual_target = camera.update(moved, .02)
        assert math.dist(actual_eye[:2], moved[:2]) == pytest.approx(3.)
        assert actual_eye[2] == pytest.approx(eye[2])
        assert actual_target[2] == pytest.approx(target[2])
    assert actual_eye[:2] == pytest.approx([5., -6.], abs=1e-4)


def test_follow_camera_turns_smoothly_across_yaw_wrap_and_holds_during_fall():
    pose = [0., 0., .8, 0., 0., math.sin(math.radians(179)/2), math.cos(math.radians(179)/2)]
    direction = [math.cos(math.radians(179)), math.sin(math.radians(179))]
    eye, target = runtime.episode_camera_view(pose, direction, distance=3., height=2., lookahead=1.5)
    camera = runtime.StableFollowCamera(pose, eye, target, .35)
    initial_heading = camera.heading
    pose[5:] = [math.sin(math.radians(-179)/2), math.cos(math.radians(-179)/2)]
    camera.update(pose, .02)
    assert 0 < camera.heading - initial_heading < math.radians(2.)
    heading = camera.heading
    pose[3:] = [1., 0., 0., 0.]  # upside down: do not orbit with the tumble
    camera.update(pose, .1, float("nan"))
    assert camera.heading == heading and camera.anchor_z == .8


def test_follow_camera_keeps_collision_calibrated_initial_distance():
    pose = [0., 0., .8, 0., 0., 0., 1.]
    camera = runtime.StableFollowCamera(pose, [-1.2, 0., 1.5], [.18, 0., .8], .35)
    eye, target = camera.update([2., 1., 2., 0., 0., 0., 1.], .1)
    assert math.dist(eye[:2], [2., 1.]) == pytest.approx(1.2)
    assert eye[2] == 1.5 and target[2] == .8


def test_follow_runtime_tracks_selected_clone_and_fixed_mode_does_not_move_camera():
    from types import SimpleNamespace
    pipeline = runtime.EnvironmentTensorPipeline.__new__(runtime.EnvironmentTensorPipeline)
    pipeline.follow_camera = None
    pipeline.agent = SimpleNamespace(get_feedback=lambda: pytest.fail("fixed mode must not follow"))
    pipeline._update_follow_camera()
    pipeline.follow_camera = runtime.StableFollowCamera([20., 0., .8, 0., 0., 0., 1.],
                                                       [17., 0., 2.8], [21., 0., .8], .35)
    pipeline.camera_env_id = 1
    pipeline.camera_origin = [20., 0., 0.]
    pipeline.camera_last_step = 0
    pipeline.pipeline_step = 5
    pipeline.ground_sampler = None
    pipeline.args = SimpleNamespace(camera_bounds_min=[-10., -10., -1.],
                                    camera_bounds_max=[10., 10., 5.], camera_clearance=.18)
    pipeline.agent = SimpleNamespace(get_feedback=lambda: {"root_pose_w": torch.tensor([
        [0., 0., .8, 0., 0., 0., 1.], [22., 1., .2, 0., 0., 0., 1.]])})
    views = []
    pipeline._set_camera_view = lambda eye, target: views.append((eye, target))
    pipeline._update_follow_camera()
    assert views[0][0] == pytest.approx([19., 1., 2.8])
    assert views[0][1] == pytest.approx([23., 1., .8])
    assert pipeline.camera_last_step == 5


def test_follow_camera_retracts_at_scene_bounds_and_smoothly_recovers():
    camera = runtime.StableFollowCamera([0., 0., .8, 0., 0., 0., 1.],
                                       [-3., 0., 2.8], [1., 0., .8], .35)
    bounds = dict(bounds_min=[-4., -4., -.5], bounds_max=[4., 4., 3.], dt=.1)
    anchor = [-3.5, 0., .95]
    eye = camera.constrain([-6.5, 0., 2.8], anchor, **bounds)
    assert eye[0] >= -3.82
    shortened = camera.safe_fraction
    assert 0 < shortened < .2
    recovered = camera.constrain([-3., 0., 2.8], [0., 0., .95], **bounds)
    assert shortened < camera.safe_fraction < 1.
    assert -3. < recovered[0] < 0.
    # Even a robot outside the bounds cannot drag the camera outside.
    eye = camera.constrain([-15., 0., 9.], [-12., 0., 7.], **bounds)
    assert -3.82 <= eye[0] <= 3.82 and -.32 <= eye[2] <= 2.82


def test_follow_camera_stays_inside_walls_and_roof_while_turning():
    import numpy as np
    wall = [[[-1., -4., -.5], [-1., 4., -.5], [-1., 4., 1.5]],
            [[-1., -4., -.5], [-1., 4., 1.5], [-1., -4., 1.5]]]
    roof = [[[-4., -4., 1.5], [4., -4., 1.5], [4., 4., 1.5]],
            [[-4., -4., 1.5], [4., 4., 1.5], [-4., 4., 1.5]]]
    ground = runtime.GroundSurfaceSampler(np.asarray(wall + roof), -.5)
    camera = runtime.StableFollowCamera([0., 0., .3, 0., 0., 0., 1.],
                                       [-.5, 0., 1.], [.1, 0., .3], .35)
    for angle in np.linspace(-math.pi, math.pi, 40):
        anchor = [0., 0., .45]
        eye = camera.constrain([3 * math.cos(angle), 3 * math.sin(angle), 2.3], anchor,
                               bounds_min=ground.bounds_min, bounds_max=ground.bounds_max,
                               segment_fraction=ground.camera_segment_fraction, dt=.1)
        assert eye[0] > -1. and eye[2] < 1.5
        assert ground.camera_segment_fraction(anchor, eye) == pytest.approx(1.)


@pytest.mark.parametrize("environment", ["marble", "gpt-v4", "marble-full"])
def test_video_switch_forwards_for_every_environment(environment):
    parser = launcher.build_parser()
    assert parser.parse_args([]).record_video is False
    args = parser.parse_args(["--environment", environment, "--record-video", "--video-fps", "25"])
    launcher.validate_args(args, parser)
    command = launcher.build_runtime_command(args)
    assert "--record-video" in command
    assert command[command.index("--video-fps") + 1] == "25.0"
    assert command[command.index("--video-dir") + 1] == "videos"


def test_video_pair_encodes_synchronized_rgb_frames_and_finalizes_mp4(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    cv2 = pytest.importorskip("cv2")
    module = load_module("walk_video", "tools/environment_walk_video.py")
    args = SimpleNamespace(video_dir=tmp_path, video_fps=None, render_every=5,
                           environment="marble-full", camera_mode="follow")
    recorder = module.WalkVideoRecorder(args)
    assert recorder.fps == 10.
    frame = np.zeros((65, 97, 3), dtype=np.uint8)
    recorder.append(frame, None)
    assert recorder.frames == 0 and not recorder.paths
    for index in range(12):
        frame[:] = [200, index * 10, 0]
        recorder.append(frame, frame[..., ::-1])
    recorder.close()
    recorder.close()
    assert recorder.frames == 12 and set(recorder.paths) == {"mesh", "gs"}
    for name, path in recorder.paths.items():
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened() and capture.get(cv2.CAP_PROP_FRAME_COUNT) == 12
        assert capture.get(cv2.CAP_PROP_FPS) == 10.
        ok, decoded = capture.read()
        assert ok and decoded.shape == (66, 98, 3)
        # RGB -> BGR conversion must preserve mesh-red / GS-blue colors.
        assert decoded[..., 2 if name == "mesh" else 0].mean() > 180
        capture.release()
    with pytest.raises(RuntimeError, match="closed"):
        recorder.append(frame, frame)


@pytest.mark.parametrize("fps", ["0", "-1", "nan", "inf"])
def test_invalid_video_fps_is_rejected_before_kit(fps):
    parser = launcher.build_parser()
    with pytest.raises(SystemExit):
        launcher.validate_args(parser.parse_args(["--video-fps", fps]), parser)


def test_video_launcher_waits_for_finalization_on_ctrl_c(monkeypatch):
    import signal
    from types import SimpleNamespace
    parser = launcher.build_parser()
    args = parser.parse_args(["--record-video"])
    launcher.validate_args(args, parser)
    waits, signals = [], []
    def wait():
        waits.append(True)
        if len(waits) == 1:
            raise KeyboardInterrupt
        return 0
    process = SimpleNamespace(pid=12345, wait=wait, poll=lambda: None)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(launcher.os, "name", "posix")
    monkeypatch.setattr(launcher.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(KeyboardInterrupt):
        launcher.run(args)
    assert len(waits) == 2 and signals == [(12345, signal.SIGINT)]
