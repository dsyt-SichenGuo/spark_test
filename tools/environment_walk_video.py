"""MP4 recording of rendered walk views, with paired mesh/GS frame submission."""
from datetime import datetime
from pathlib import Path
import os

import numpy as np


class WalkVideoRecorder:
    def __init__(self, args):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video recording requires OpenCV: python -m pip install opencv-python-headless") from exc
        self.cv2 = cv2
        self.fps = args.video_fps or 50. / args.render_every
        self.directory = args.video_dir.expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = f"{args.environment}_{args.camera_mode}_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}"
        self.names = ("mesh", "gs") if args.environment == "marble-full" else ("mesh",)
        self.writers, self.paths, self.sizes = {}, {}, {}
        self.frames = 0
        self.closed = False
        print(f"[Video] enabled: {self.directory}, views={self.names}, FPS={self.fps:g}", flush=True)

    def append(self, mesh_rgb, gs_rgb=None):
        if self.closed:
            raise RuntimeError("Video recorder is closed")
        images = {"mesh": mesh_rgb, "gs": gs_rgb}
        # Do not write either view unless the synchronized pair is ready.
        if any(images[name] is None for name in self.names):
            return
        images = {name: np.asarray(images[name])[..., :3] for name in self.names}
        if any(image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 for image in images.values()):
            raise ValueError("Video frames must be uint8 RGB images")
        for name, image in images.items():
            height, width = image.shape[:2]
            if name not in self.writers:
                # MPEG-4 needs even dimensions. Pad one edge if necessary.
                size = (width + width % 2, height + height % 2)
                path = self.directory / f"{self.prefix}_{name}.mp4"
                writer = self.cv2.VideoWriter(str(path), self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Cannot open MP4 encoder for {path}")
                self.writers[name], self.paths[name], self.sizes[name] = writer, path, (width, height)
            if self.sizes[name] != (width, height):
                raise ValueError("Recording resolution changed during the run")
        for name, image in images.items():
            height, width = image.shape[:2]
            bgr = np.ascontiguousarray(image[..., ::-1])
            if width % 2 or height % 2:
                bgr = np.pad(bgr, ((0, height % 2), (0, width % 2), (0, 0)), mode="edge")
            self.writers[name].write(bgr)
        self.frames += 1

    def close(self):
        if self.closed:
            return
        self.closed = True
        for writer in self.writers.values():
            writer.release()
        for name, path in self.paths.items():
            print(f"[Video] saved {name}: {path} ({self.frames} frames, {self.frames / self.fps:.2f}s)", flush=True)
        if not self.frames:
            print("[Video] no rendered frames captured; no MP4 produced", flush=True)


class NativeWalkVideoCapture:
    """Capture the task/active viewport camera without changing its UI or lighting."""
    def __init__(self):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom
        self.stage = omni.usd.get_context().get_stage()
        self.camera = UsdGeom.Camera.Define(self.stage, "/World/StaticWalkTask/RecordingCamera")
        self.transform = self.camera.AddTransformOp()
        self.camera.CreateClippingRangeAttr((.01, 1000.))
        self.product = rep.create.render_product(str(self.camera.GetPath()), (960, 720))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb.attach([self.product])

    def prepare(self, headless):
        from pxr import UsdGeom
        path = "/World/StaticWalkTask/TaskCamera"
        if not headless:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                path = str(viewport.camera_path)
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
        if not camera:
            raise RuntimeError(f"Recording camera not found: {path}")
        self.transform.Set(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0))
        self.camera.GetFocalLengthAttr().Set(camera.GetFocalLengthAttr().Get())
        aperture = camera.GetHorizontalApertureAttr().Get()
        self.camera.GetHorizontalApertureAttr().Set(aperture)
        self.camera.GetVerticalApertureAttr().Set(aperture * .75)

    def frame(self):
        image = np.asarray(self.rgb.get_data())
        return image if image.ndim == 3 and image.shape[:2] == (720, 960) else None

    def close(self):
        self.rgb.detach([self.product])
        self.product.destroy()
