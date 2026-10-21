"""Bounded-memory MP4 recording of the displayed GS/mesh pair in wall time."""

from datetime import datetime, timezone
from pathlib import Path
import math
import threading
import time

import numpy as np


class PairedVideoRecorder:
    """Encode the latest published pair at a fixed wall-clock frame rate.

    Only one immutable BGR image is retained. The encoder repeats that image
    between display updates; simulation speed and GS latency do not shorten
    the recording. A new file is created for every start/stop interval.
    """

    def __init__(self, directory, fps=30.):
        if not math.isfinite(fps) or not 1 <= fps <= 120:
            raise ValueError("Video FPS must be finite and in [1, 120]")
        self.directory = Path(directory).expanduser().resolve()
        self.fps = float(fps)
        self._condition = threading.Condition()
        self._latest = None
        self._thread = None
        self._started = self._stopped = None
        self._error = None
        self.path = self.last_path = None
        self.frames = 0
        self.last_duration = 0.

    @property
    def ready(self):
        return self._latest is not None

    @property
    def active(self):
        return self._thread is not None and self._stopped is None and self._error is None

    @property
    def error(self):
        return self._error

    @property
    def elapsed(self):
        if self._started is None:
            return 0.
        end = self._stopped if self._stopped is not None else time.monotonic()
        return max(0., end - self._started)

    def update(self, gs_rgb, mesh_rgb):
        gs, mesh = np.asarray(gs_rgb), np.asarray(mesh_rgb)
        if (gs.ndim != 3 or mesh.ndim != 3 or gs.shape[:2] != mesh.shape[:2]
                or gs.shape[2] not in (3, 4) or mesh.shape[2] not in (3, 4)
                or gs.dtype != np.uint8 or mesh.dtype != np.uint8):
            raise ValueError("Recording requires a matching pair of uint8 RGB/RGBA images")
        # MP4 codecs require even dimensions. Pad only the outer right/bottom
        # edge; the GS/mesh split and original pixel coordinates are preserved.
        rgb = np.concatenate((gs[..., :3], mesh[..., :3]), axis=1)
        height, width = rgb.shape[:2]
        if height % 2 or width % 2:
            rgb = np.pad(rgb, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        with self._condition:
            if self._thread is not None and self._latest.shape != bgr.shape:
                raise ValueError("Recording resolution cannot change within a clip")
            self._latest = bgr

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Stop the current recording before starting another")
        if not self.ready:
            raise RuntimeError("Wait for the first synchronized GS/mesh frame before recording")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Recording requires OpenCV in the active Python environment") from exc
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        path = self.directory / f"gs_mesh_{stamp}.mp4"
        height, width = self._latest.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            writer.release()
            path.unlink(missing_ok=True)
            raise RuntimeError(f"Cannot open the MP4 encoder: {path}")
        self.path = path
        self.frames = 0
        self._error = self._stopped = None
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._encode, args=(writer,),
                                        name="GS-Mesh-MP4", daemon=True)
        self._thread.start()
        print(f"[video] Recording: {path} | GS left, mesh right | {width}x{height} @ {self.fps:g} FPS", flush=True)
        return path

    def _encode(self, writer):
        try:
            index = 0
            while True:
                with self._condition:
                    deadline = self._started + index / self.fps
                    if self._stopped is not None and deadline >= self._stopped:
                        break
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        self._condition.wait(delay)
                        continue
                    frame = self._latest
                writer.write(frame)
                self.frames += 1
                index += 1
        except Exception as exc:
            self._error = exc
        finally:
            writer.release()

    def stop(self):
        if self._thread is None:
            return self.last_path
        with self._condition:
            self._stopped = time.monotonic()
            self._condition.notify_all()
        self._thread.join()
        self._thread = None
        self.last_duration = self.elapsed
        error, self._error = self._error, None
        if error is not None:
            raise RuntimeError(f"MP4 encoding failed for {self.path}: {error}") from error
        if self.frames:
            self.last_path = self.path
            print(f"[video] Saved: {self.path} | {self.frames} frames | "
                  f"{self.frames / self.fps:.2f}s video / {self.last_duration:.2f}s recording", flush=True)
        else:
            self.path.unlink(missing_ok=True)
            self.last_path = None
            print("[video] No frames recorded; no MP4 produced", flush=True)
        return self.last_path
