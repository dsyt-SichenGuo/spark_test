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


@pytest.mark.parametrize("environment", ["marble", "gpt-v4"])
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


@pytest.mark.parametrize("environment", ["marble", "gpt-v4"])
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


@pytest.mark.parametrize("environment", ["marble", "gpt-v4"])
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
