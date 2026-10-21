"""Exercise actual MP4 encoding and manual recording of synchronized frames."""

from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_walk.recording import PairedVideoRecorder
from reconstruction_walk.rendering import DualViewRenderer


def decode(path):
    cv2 = pytest.importorskip("cv2")
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame[..., ::-1])
    finally:
        capture.release()
    return fps, frames


def test_mp4_order_wall_time_and_repeated_start_stop(tmp_path):
    pytest.importorskip("cv2")
    recorder = PairedVideoRecorder(tmp_path, fps=20.)
    # Odd input height must be padded without losing edge pixels.
    gs = np.full((65, 67, 4), [255, 0, 0, 255], np.uint8)
    mesh = np.full((65, 67, 4), [0, 0, 255, 255], np.uint8)
    recorder.update(gs, mesh)
    assert not recorder.active and not list(tmp_path.glob("*.mp4"))
    path = recorder.start()
    try:
        # No new display frames: the last displayed pair must fill wall time.
        time.sleep(.18)
    finally:
        recorder.stop()
    fps, frames = decode(path)
    assert fps == 20. and len(frames) >= 4
    assert abs(len(frames) / fps - recorder.last_duration) <= 1 / fps + .005
    assert len(frames) == recorder.frames
    assert frames[0].shape == (66, 134, 3)
    assert np.allclose(frames[0][10:50, 10:50].mean(axis=(0, 1)), [255, 0, 0], atol=8)
    assert np.allclose(frames[0][10:50, 80:120].mean(axis=(0, 1)), [0, 0, 255], atol=8)
    assert not recorder.active and recorder.last_path == path
    assert recorder.stop() == path

    recorder.update(mesh, gs)
    second = recorder.start()
    time.sleep(.06)
    recorder.stop()
    assert second != path and second.is_file()
    _, second_frames = decode(second)
    assert np.allclose(second_frames[0][10:50, 10:50].mean(axis=(0, 1)), [0, 0, 255], atol=8)
    assert len(list(tmp_path.glob("*.mp4"))) == 2


def test_manual_request_waits_for_pair_and_stops_at_user_action(tmp_path):
    pytest.importorskip("cv2")
    renderer = object.__new__(DualViewRenderer)
    renderer.recorder = PairedVideoRecorder(tmp_path, fps=20.)
    renderer.record_requested = False
    renderer.record_button = SimpleNamespace(text="")
    renderer.video_status = SimpleNamespace(text="", tooltip="")
    renderer.mesh_status = SimpleNamespace(text="")
    renderer.gs_status = SimpleNamespace(text="")
    renderer.mesh_provider = renderer.gs_provider = SimpleNamespace(set_data_array=lambda *args: None)
    renderer.width = renderer.height = 64
    renderer.saved = True
    renderer.toggle_recording()
    assert renderer.record_requested and not renderer.recorder.active
    assert not list(tmp_path.glob("*.mp4"))
    mesh = np.full((64, 64, 4), [0, 0, 255, 255], np.uint8)
    gs = np.full((64, 64, 4), [255, 0, 0, 255], np.uint8)
    renderer._publish({"rgb": mesh, "sequence": 1, "policy": "wbt", "time": 0.}, gs)
    assert renderer.recorder.active and not renderer.record_requested
    time.sleep(.06)
    renderer.toggle_recording()
    assert not renderer.recorder.active
    assert renderer.video_status.text.startswith("Saved:")
    assert renderer.record_button.text.startswith("Start recording")
    frames = renderer.recorder.frames
    renderer._publish({"rgb": gs, "sequence": 2, "policy": "wbt", "time": .1}, mesh)
    assert renderer.recorder.frames == frames
    assert len(decode(renderer.recorder.last_path)[1]) == frames


def test_recorder_rejects_unpaired_images(tmp_path):
    recorder = PairedVideoRecorder(tmp_path)
    with pytest.raises(RuntimeError, match="synchronized"):
        recorder.start()
    with pytest.raises(ValueError, match="matching pair"):
        recorder.update(np.zeros((64, 64, 4), np.uint8), np.zeros((65, 64, 4), np.uint8))
    assert not list(tmp_path.glob("*.mp4"))
