"""CPU checks for geometry, keyboard/fall behavior and frame-consistent compositing."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_walk.geometry import calibrate, cluster_mesh, SurfaceQueries, similarity
from reconstruction_walk.task import KeyboardWalkTask
from reconstruction_walk.rendering import composite, robot_pixel_mask


def test_keyboard_release_and_opposed_keys():
    task = KeyboardWalkTask()
    task.age = 100
    task.keys.update(["W", "S", "A"])
    cmd = task.compute_command()
    assert cmd[0] == 0 and cmd[2] > 0
    task.keys.clear()
    for _ in range(100):
        cmd = task.compute_command()
    assert np.allclose(cmd, 0)
    task.keys.add("W")
    assert task.compute_command()[0] > 0
    task.clear_input()
    assert np.allclose(task.compute_command(), 0)


@pytest.mark.parametrize("key_format", ["enum", "string", "lowercase", "qualified"])
def test_keyboard_callback_handles_enum_and_string_inputs(monkeypatch, key_format):
    from types import ModuleType, SimpleNamespace
    from reconstruction_walk.runtime import ReconstructionWalkPipeline

    carb = ModuleType("carb")
    carb.input = ModuleType("carb.input")
    types = SimpleNamespace(KEY_PRESS=0, KEY_REPEAT=1, KEY_RELEASE=2, CHAR=3)
    carb.input.KeyboardEventType = types
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.input", carb.input)
    pipeline = object.__new__(ReconstructionWalkPipeline)
    pipeline.task = KeyboardWalkTask()
    recording_toggles = []
    pipeline.renderer = SimpleNamespace(toggle_recording=lambda: recording_toggles.append(True))

    def key(value):
        if key_format == "enum":
            return SimpleNamespace(name=value)
        if key_format == "lowercase":
            return value.lower()
        if key_format == "qualified":
            return "KeyboardInput." + value
        return " " if value == "SPACE" else value

    def send(kind, value):
        assert pipeline.key_event(SimpleNamespace(type=kind, input=value)) is True

    for name in ("W", "A", "S", "D", "Q", "E"):
        send(types.KEY_PRESS, key(name))
        assert name in pipeline.task.keys
        send(types.KEY_RELEASE, key(name))
        assert name not in pipeline.task.keys

    # Text input and key repeats must not trigger movement/reset, and must be
    # filtered before accessing input (which need not even exist on other events).
    send(types.CHAR, "r")
    send(types.CHAR, "w")
    assert pipeline.key_event(SimpleNamespace(type=types.KEY_REPEAT)) is True
    assert not pipeline.task.reset_requested and not pipeline.task.keys
    send(types.KEY_PRESS, None)
    send(types.KEY_PRESS, key("ESCAPE"))
    assert not pipeline.task.keys

    pipeline.task.age = 100
    send(types.KEY_PRESS, key("W"))
    assert pipeline.task.compute_command()[0] > 0
    send(types.KEY_PRESS, key("SPACE"))
    assert pipeline.task.keys == {"SPACE"}
    assert np.allclose(pipeline.task.command, 0)
    send(types.KEY_RELEASE, key("SPACE"))
    assert not pipeline.task.keys
    send(types.KEY_PRESS, key("R"))
    assert pipeline.task.reset_requested
    send(types.KEY_PRESS, key("F9"))
    send(types.KEY_RELEASE, key("F9"))
    send(types.KEY_REPEAT, key("F9"))
    assert len(recording_toggles) == 1


def test_fall_confirmation_and_nearby_respawn():
    task = KeyboardWalkTask(fall_confirm_steps=3)
    task.age = 100
    fallen = [1., 2., .1, 0., 0., 0., 1.]
    assert task.observe(fallen, 0) is None
    assert task.observe(fallen, 0) is None
    assert task.observe(fallen, 0) == "fallen"
    candidates = [[0., 0., 0., 0.], [1., 2.1, .1, .5], [8., 8., 0., 0.]]
    assert np.allclose(task.choose_spawn(candidates, [1, 2, .1]), candidates[1])
    task.last_safe = np.array([8, 8, .8])
    assert np.allclose(task.choose_spawn(candidates, [30, 30, .1]), candidates[2])
    task.reset()
    assert task.fall_count == task.age == 0 and not task.keys


def test_calibration_recovers_rotated_scaled_floor():
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(1)
    # Many disconnected floor triangles suffice for plane fitting; no floor is
    # manufactured by the actual preparation pipeline.
    centers = rng.uniform([-3, -3, 0], [3, 3, 0], size=(400, 3))
    tri = centers[:, None] + np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]])
    rotation = Rotation.from_euler("xyz", [.6, -.4, .3]).as_matrix()
    translation = np.array([4., 2., -3.])
    native = (tri.reshape(-1, 3) @ rotation.T) * 2 + translation
    faces = np.arange(len(native)).reshape(-1, 3)
    cameras = np.array([[0., 0., 1.55], [1., 0., 1.55], [2., 0., 1.55]]) @ rotation.T * 2 + translation
    # Camera CV axes: right=-Y, down=-Z, forward=X.
    cv = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    rotations = np.repeat((rotation @ cv)[None], 3, axis=0)
    s, r, t, _ = calibrate(native, faces, cameras, rotations)
    assert np.isclose(s, .5, atol=1e-5)
    world = native @ r.T * s + t
    assert np.max(np.abs(world[:, 2])) < 1e-5


def test_surface_queries_reject_holes_and_preserve_empty_regions():
    vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    points, triangles = cluster_mesh(vertices, faces, .05)
    query = SurfaceQueries(points, triangles)
    height = query.heights([[0, 0], [3, 0]])
    assert np.isclose(height[0], 0) and np.isnan(height[1])


def test_instance_filter_and_depth_occlusion():
    native = np.full((1, 3, 4), 200, np.uint8)
    gs = np.full((1, 3, 4), 50, np.uint8)
    gs[:, :, 3] = 255
    instances = {"data": np.array([[1, 1, 2]]), "info": {"idToLabels": {
        "1": "/World/envs/env_0/Robot/torso", "2": "/World/WalkEnvironment/Visual/Surface"}}}
    mask = robot_pixel_mask(instances, (1, 3))
    image = composite(gs, np.array([[3., 1., 3.]]), native, np.array([[2., 2., 2.]]), mask)
    assert image[0, 0, 0] == 200  # Robot in front of Gaussian surface.
    assert image[0, 1, 0] == 50   # Robot hidden by Gaussian wall.
    assert image[0, 2, 0] == 50   # Never paste the native environment into GS.


def test_invalid_similarities_rejected():
    with pytest.raises(ValueError):
        similarity({"scale": -1})
    with pytest.raises(ValueError):
        similarity({"rotation": np.diag([1, 1, -1])})


def test_follow_camera_accepts_actual_float32_robot_feedback():
    from pxr import Gf
    from types import SimpleNamespace
    from reconstruction_walk.rendering import DualViewRenderer
    renderer = object.__new__(DualViewRenderer)
    renderer.Gf = Gf
    renderer.eye = None
    renderer.last_yaw = 0.
    renderer.query = SimpleNamespace(camera_eye=lambda target, eye: eye)
    matrices = []
    renderer.transform = SimpleNamespace(Set=matrices.append)
    pose = renderer.camera_pose(np.array([1., 2., .8, 0., 0., 0., 1.], dtype=np.float32))
    assert np.isfinite(pose).all() and np.allclose(pose[3], [0, 0, 0, 1])
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3))
    assert len(matrices) == 1


def test_invalid_scene_scale_is_rejected_before_geometry(tmp_path):
    import json
    from reconstruction_walk.config import SceneConfig
    for name in ("gs.ply", "mesh.ply", "cameras.json"):
        (tmp_path / name).write_text("")
    data = {"version": 1, "gaussian": {"path": "gs.ply"}, "mesh": {"path": "mesh.ply"},
            "cameras": "cameras.json", "camera_height_m": 1.55, "meters_per_unit": float("nan")}
    config = tmp_path / "scene.json"
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="meters_per_unit"):
        SceneConfig.read(config)
