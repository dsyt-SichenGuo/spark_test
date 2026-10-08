"""Original SPZ appearance composited with native Isaac RGB and camera-Z depth.

No source USD stages or people are imported. The Gaussian-to-collider transform
matches the original GLB conversion used to author /World/MarbleCollider.
Isaac/Kit imports are deferred so decoding and composition can be tested offline.
"""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time

import numpy as np


def configure_cuda_toolkit():
    """Support NVIDIA Conda's targets/ header layout without changing the environment."""
    prefix = Path(sys.prefix)
    binary = str(prefix / "bin")
    if binary not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")
    include = prefix / "targets/x86_64-linux/include"
    library = prefix / "targets/x86_64-linux/lib"
    if (prefix / "bin/nvcc").is_file() and include.is_dir():
        os.environ.setdefault("CUDA_HOME", str(prefix))
        extension = sys.modules.get("torch.utils.cpp_extension")
        if extension is not None and extension.CUDA_HOME is None:
            extension.CUDA_HOME = os.environ["CUDA_HOME"]
        for name, directory in (("CPATH", include), ("LIBRARY_PATH", library)):
            paths = os.environ.get(name, "").split(os.pathsep)
            if str(directory) not in paths:
                os.environ[name] = os.pathsep.join([str(directory), *(p for p in paths if p)])
    os.environ.setdefault("MAX_JOBS", "4")


def load_spz(path):
    """Read SPZ v2 degree-zero Gaussian means, rotations, scales, RGB and alpha."""
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 16:
        raise ValueError("Truncated SPZ header")
    magic, version, count, degree, bits, flags, _ = struct.unpack_from("<III4B", data)
    if (magic, version, degree, flags) != (0x5053474E, 2, 0, 0):
        raise ValueError("Expected the bundled SPZ v2 degree-zero asset")
    if count == 0 or len(data) != 16 + count * 19:
        raise ValueError("Invalid SPZ payload size")
    offset = 16

    def take(size):
        nonlocal offset
        value = np.frombuffer(data, np.uint8, size, offset)
        offset += size
        return value

    packed = take(count * 9).reshape(count, 3, 3).astype(np.int32)
    xyz = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    means = np.where(xyz & 0x800000, xyz - 0x1000000, xyz).astype(np.float32) / (1 << bits)
    alpha = take(count).astype(np.float32) / 255
    dc = (take(count * 3).reshape(count, 3).astype(np.float32) / 255 - .5) / .15
    colors = np.clip(dc * .28209479177387814 + .5, 0, 1)
    scales = np.exp(take(count * 3).reshape(count, 3).astype(np.float32) / 16 - 10)
    xyz = take(count * 3).reshape(count, 3).astype(np.float32) / 127.5 - 1
    w = np.sqrt(np.maximum(0, 1 - np.sum(xyz * xyz, axis=1, keepdims=True)))
    quats = np.concatenate([w, xyz], axis=1)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return means, quats, scales, colors, alpha


def configure_gaussian_architecture(torch, device):
    """Repair invalid inherited architecture flags using the selected GPU."""
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    capability = torch.cuda.get_device_capability(device)
    detected = f"{capability[0]}.{capability[1]}"
    configured = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if not configured:
        os.environ["TORCH_CUDA_ARCH_LIST"] = detected
    try:
        _get_cuda_arch_flags()
    except ValueError:
        os.environ["TORCH_CUDA_ARCH_LIST"] = detected
        # Validate the fallback too; do not claim support for unsupported hardware.
        _get_cuda_arch_flags()
        print(f"[Marble full] invalid inherited TORCH_CUDA_ARCH_LIST={configured!r}; "
              f"using selected GPU capability {detected} for this process.", flush=True)


def preflight_gaussian_renderer(device):
    """Compile and exercise CUDA before Kit owns the GUI event loop."""
    configure_cuda_toolkit()
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("marble-full visualization requires a working CUDA device")
    torch.cuda.set_device(device)
    # Keep auto-detected architecture stable across preflight and Kit startup.
    configure_gaussian_architecture(torch, device)
    started = time.monotonic()
    done = threading.Event()

    def progress():
        while not done.wait(15):
            print(f"[Marble full] CUDA preflight still running ({time.monotonic() - started:.0f}s); "
                  "first-time compilation can take several minutes. Isaac window opens afterwards.", flush=True)

    print("[Marble full] checking/compiling gsplat CUDA before opening Isaac Sim...", flush=True)
    reporter = threading.Thread(target=progress, daemon=True)
    reporter.start()
    try:
        from gsplat import rasterization
        with torch.inference_mode():
            rgbd, _, _ = rasterization(
                torch.tensor([[0., 0., 2.]], device=device),
                torch.tensor([[1., 0., 0., 0.]], device=device),
                torch.full((1, 3), .1, device=device), torch.ones(1, device=device),
                torch.ones((1, 3), device=device), torch.eye(4, device=device)[None],
                torch.tensor([[[24., 0., 16.], [0., 24., 16.], [0., 0., 1.]]], device=device),
                32, 32, packed=True, render_mode="RGB+ED",
            )
            if not bool(torch.isfinite(rgbd).all().item()):
                raise RuntimeError("gsplat CUDA preflight produced non-finite pixels")
    finally:
        done.set()
        reporter.join()
    print(f"[Marble full] CUDA ready in {time.monotonic() - started:.1f}s; starting Isaac Sim.", flush=True)


def gaussian_to_collider_transform(alignment):
    """Column-vector similarity: source RUB -> Z-up -> asset-local collider."""
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    translation = np.asarray(alignment["translation"], dtype=np.float64)
    scale = float(alignment["scale_marble_units_per_video_meter"])
    if (rotation.shape != (3, 3) or translation.shape != (3,) or scale <= 0
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not math.isfinite(scale)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.)):
        raise ValueError("Invalid Gaussian-to-collider alignment")
    rub_to_zup = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    return rotation.T @ rub_to_zup, -rotation.T @ translation / scale, 1. / scale


def compose_depth(gaussian_rgb, gaussian_depth, gaussian_alpha, native_rgb, native_depth):
    """Select the nearest surface; both depth buffers use the same camera-Z units."""
    native_depth = np.asarray(native_depth).reshape(gaussian_depth.shape)
    native_valid = np.isfinite(native_depth) & (native_depth > .01) & (native_depth < 1000)
    gaussian_valid = np.isfinite(gaussian_depth) & (gaussian_depth > .01) & (gaussian_alpha > .2)
    visible = native_valid & (~gaussian_valid | (native_depth < gaussian_depth))
    rgb = np.asarray(gaussian_rgb).copy()
    rgb[visible] = np.asarray(native_rgb)[..., :3][visible]
    return rgb, {"native_pixels": int(native_valid.sum()), "visible_native_pixels": int(visible.sum()),
                 "occluded_native_pixels": int((native_valid & ~visible).sum())}


def exclude_environment_depth(native_depth, instances):
    """Keep mesh visible in the main viewport, but use GS appearance in the inset."""
    ids = np.asarray(instances["data"])
    depth = np.asarray(native_depth).reshape(ids.shape).copy()
    labels = instances["info"]["idToLabels"]
    environment_ids = [int(key) for key, path in labels.items()
                       if isinstance(path, str) and path.startswith("/World/StaticWalkTask/env_")
                       and "/Scene/Environment" in path]
    mask = np.isin(ids, environment_ids)
    depth[mask] = np.inf
    return depth, int(mask.sum())


class MarbleGaussianCompositor:
    """A separate camera render product plus an in-process gsplat renderer."""

    def __init__(self, args, asset, device):
        configure_cuda_toolkit()
        import torch
        if torch.device(device).type != "cuda":
            raise ValueError("marble-full visualization requires a CUDA device")
        from gsplat import rasterization
        from scipy.spatial.transform import Rotation
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom
        from scrpits.run_unitree_g1_usd_environment_walk_benchmark import GAUSSIAN_FILES

        self.torch, self.rasterization = torch, rasterization
        self.device, self.args = device, args
        self.width, self.height = args.gaussian_resolution
        self.stage = omni.usd.get_context().get_stage()
        self.camera = UsdGeom.Camera.Define(self.stage, "/World/StaticWalkTask/CompositeCamera")
        self.camera.CreateClippingRangeAttr((.01, 1000.))
        self.camera.CreateFocalLengthAttr(18.)
        self.camera.CreateHorizontalApertureAttr(20.955)
        self.camera.CreateVerticalApertureAttr(20.955 * self.height / self.width)
        self.transform = self.camera.AddTransformOp()
        self.origin = np.zeros(3)
        self.gui = not args.headless
        self.last_image, self.last_native_image, self.last_stats = None, None, {}
        self.frames = 0
        self.gaussian_renders = 0
        self._cached_view = self._cached_gaussians = None
        self.window = None
        path = asset["gaussian_dir"] / GAUSSIAN_FILES[args.gaussian_quality]
        means, quats, scales, colors, alpha = load_spz(path)
        alignment = json.loads(asset["gaussian_alignment"].read_text())
        rotation, shift, scale = gaussian_to_collider_transform(alignment)
        means = means @ rotation.T * scale + shift
        means[:, 2] += asset["world_z_offset"]
        # Left-multiply source wxyz quaternions by the coordinate-system rotation.
        q = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]].astype(np.float32)
        qw = q[0] * quats[:, :1] - np.sum(quats[:, 1:] * q[1:], axis=1, keepdims=True)
        qxyz = q[0] * quats[:, 1:] + quats[:, :1] * q[1:] + np.cross(q[1:], quats[:, 1:])
        quats = np.concatenate([qw, qxyz], axis=1)
        self.means, self.quats, self.scales, self.colors, self.alpha = [
            torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
            for a in (means, quats, scales * scale, colors, alpha)
        ]
        # The tiny CUDA preflight has already run before Kit startup.
        torch.cuda.set_device(device)
        self.product = rep.create.render_product(str(self.camera.GetPath()), (self.width, self.height))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.instances = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation",
                                                           init_params={"colorize": False})
        self.rgb.attach([self.product])
        self.depth.attach([self.product])
        self.instances.attach([self.product])
        if self.gui:
            import omni.ui as ui
            self.provider = ui.ByteImageProvider()
            self.window = ui.Window("Marble full appearance + G1", width=self.width, height=self.height + 60)
            with self.window.frame:
                with ui.VStack():
                    self.status = ui.Label("Preparing Gaussian + robot frame...", height=25)
                    ui.ImageWithProvider(self.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    ui.Label("Main view: Marble mesh | This view: GS + G1 | Cameras synchronized", height=25)
            # Keep the native viewport accessible for its usual camera controls.
            viewport_window = ui.Workspace.get_window("Viewport")
            if viewport_window:
                self.window.dock_in(viewport_window, ui.DockPosition.RIGHT)
        print(f"[Marble full] loaded {len(means):,} Gaussians; static collider retained; no people", flush=True)

    def set_view(self, eye, target, origin):
        from pxr import Gf
        self.origin = np.asarray(origin, dtype=np.float64)
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0., 0., 1.)).GetInverse()
        self.transform.Set(matrix)

    def prepare(self):
        from pxr import UsdGeom
        if self.gui:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                camera = UsdGeom.Camera(self.stage.GetPrimAtPath(str(viewport.camera_path)))
                if camera:
                    matrix = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)
                    if self.transform.Get() != matrix:
                        self.transform.Set(matrix)
                    for name in ("focalLength", "horizontalAperture"):
                        attribute = self.camera.GetPrim().GetAttribute(name)
                        value = camera.GetPrim().GetAttribute(name).Get()
                        if attribute.Get() != value:
                            attribute.Set(value)
                    # The composite's aspect ratio is independent of viewport size.
                    aperture = self.camera.GetHorizontalApertureAttr().Get()
                    vertical = self.camera.GetVerticalApertureAttr()
                    value = aperture * self.height / self.width
                    if not np.isclose(vertical.Get(), value):
                        vertical.Set(value)
        self.pending_c2w = np.asarray(UsdGeom.Xformable(self.camera).ComputeLocalToWorldTransform(0)).T.copy()
        self.pending_c2w[:3, 3] -= self.origin

    def _render_gaussians(self, c2w):
        torch = self.torch
        view = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(c2w)
        focal = float(self.camera.GetFocalLengthAttr().Get()) / float(self.camera.GetHorizontalApertureAttr().Get()) * self.width
        intrinsic = np.array([[focal, 0., self.width / 2], [0., focal, self.height / 2], [0., 0., 1.]])
        with torch.inference_mode():
            rgbd, alpha, _ = self.rasterization(
                self.means, self.quats, self.scales, self.alpha, self.colors,
                torch.as_tensor(view[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(intrinsic[None], dtype=torch.float32, device=self.device),
                self.width, self.height, packed=True, near_plane=.01, far_plane=1000., render_mode="RGB+ED",
            )
        return ((rgbd[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy(),
                rgbd[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy())

    def update(self):
        native = np.asarray(self.rgb.get_data())
        depth = np.asarray(self.depth.get_data())
        if native.size == 0 or depth.size == 0:
            return
        if native.shape[:2] != (self.height, self.width):
            return
        instances = self.instances.get_data()
        if not isinstance(instances, dict) or np.asarray(instances.get("data", [])).size != self.width * self.height:
            return
        depth, mesh_pixels = exclude_environment_depth(depth, instances)
        view = (self.pending_c2w.tobytes(), float(self.camera.GetFocalLengthAttr().Get()),
                float(self.camera.GetHorizontalApertureAttr().Get()))
        if view != self._cached_view:
            self._cached_gaussians = self._render_gaussians(self.pending_c2w)
            self._cached_view = view
            self.gaussian_renders += 1
        rgb, z, alpha = self._cached_gaussians
        self.last_native_image = native[..., :3].copy()
        self.last_image, self.last_stats = compose_depth(rgb, z, alpha, native, depth)
        self.last_stats["mesh_pixels_excluded_from_gs"] = mesh_pixels
        self.frames += 1
        if self.gui:
            rgba = np.concatenate([self.last_image, np.full((self.height, self.width, 1), 255, np.uint8)], axis=2)
            self.provider.set_data_array(rgba, [self.width, self.height])
            self.status.text = f"Marble + G1 | {self.frames} frames | depth occlusion enabled"
        if self.frames == 1:
            print(f"[Marble full] composite ready: {self.last_stats}", flush=True)

    def save(self, path):
        from PIL import Image
        if self.last_image is None:
            raise RuntimeError("No composite frame was captured")
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.last_image).save(path)
        print(f"[Marble full] snapshot={path}, frames={self.frames}, "
              f"gaussian_renders={self.gaussian_renders}, stats={self.last_stats}", flush=True)

    def close(self):
        self.rgb.detach([self.product])
        self.depth.detach([self.product])
        self.instances.detach([self.product])
        self.product.destroy()
        if self.window is not None:
            self.window.destroy()
