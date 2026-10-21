"""Gaussian renderer registry. All backends return RGB/A and world camera-Z depth."""

import importlib
import math
import os
import sys
from pathlib import Path
from argparse import Namespace

import numpy as np


REGISTRY = {}


def register_backend(name, backend_class):
    REGISTRY[name] = backend_class


def create_backend(cfg, alignment, device="cuda:0"):
    name = cfg.data["gaussian"]["backend"]
    if name not in REGISTRY and ":" in name:
        module, symbol = name.split(":", 1)
        register_backend(name, getattr(importlib.import_module(module), symbol))
    if name not in REGISTRY:
        raise ValueError(f"Unknown Gaussian backend {name!r}; available: {list(REGISTRY)}")
    return REGISTRY[name](cfg, alignment, device)


class NativeCoordinateRenderer:
    def __init__(self, cfg, alignment, device):
        import torch
        from .geometry import similarity
        self.torch, self.device = torch, device
        torch.cuda.set_device(device)
        # NVIDIA Conda's toolkit uses targets/*/include, while PyTorch JIT
        # extensions also need a discoverable nvcc. Do not replace user installs.
        prefix = Path(sys.prefix)
        if (prefix / "bin/nvcc").exists():
            os.environ["PATH"] = str(prefix / "bin") + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("CUDA_HOME", str(prefix))
            for key, directory in (("CPATH", prefix / "targets/x86_64-linux/include"),
                                   ("LIBRARY_PATH", prefix / "targets/x86_64-linux/lib")):
                if directory.exists():
                    os.environ[key] = str(directory) + os.pathsep + os.environ.get(key, "")
        from torch.utils import cpp_extension
        if cpp_extension.CUDA_HOME is None:
            cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME")
        try:
            cpp_extension._get_cuda_arch_flags()
        except ValueError:
            major, minor = torch.cuda.get_device_capability(device)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
            cpp_extension._get_cuda_arch_flags()
        os.environ.setdefault("MAX_JOBS", "4")
        self.scale, self.rotation, self.translation = similarity(alignment)

    def native_camera(self, world_camera):
        pose = np.asarray(world_camera, dtype=np.float64).reshape(4, 4)
        native = np.eye(4)
        native[:3, :3] = self.rotation.T @ pose[:3, :3]
        native[:3, 3] = self.rotation.T @ (pose[:3, 3] - self.translation) / self.scale
        return native

    def close(self):
        pass


class Matcha2DGSRenderer(NativeCoordinateRenderer):
    def __init__(self, cfg, alignment, device):
        super().__init__(cfg, alignment, device)
        source = cfg.asset(cfg.data["gaussian"]["renderer_source"])
        for path in (source, source / "submodules/simple-knn", source / "submodules/diff-surfel-rasterization"):
            sys.path.insert(0, str(path))
        from gaussian_renderer import render
        from scene.gaussian_model import GaussianModel
        from scene.cameras import MiniCam
        from utils.graphics_utils import getProjectionMatrix
        self.render_function, self.MiniCam, self.projection = render, MiniCam, getProjectionMatrix
        self.model = GaussianModel(3)
        self.model.load_ply(str(cfg.asset(cfg.data["gaussian"]["path"])))
        self.pipeline = Namespace(compute_cov3D_python=False, convert_SHs_python=False, depth_ratio=.5, debug=False)
        self.background = self.torch.zeros(3, dtype=self.torch.float32, device=device)
        self.count = len(self.model.get_xyz)

    def render(self, camera, intrinsics, width, height):
        torch = self.torch
        native = self.native_camera(camera)
        k = np.asarray(intrinsics)
        if not np.allclose([k[0, 2], k[1, 2]], [width / 2, height / 2], atol=.01):
            raise ValueError("MAtCha preview backend requires centered camera intrinsics")
        fovx, fovy = 2 * math.atan(width / (2 * k[0, 0])), 2 * math.atan(height / (2 * k[1, 1]))
        with torch.inference_mode():
            view = torch.as_tensor(np.linalg.inv(native).copy(), dtype=torch.float32, device=self.device).T.contiguous()
            projection = self.projection(.01 / self.scale, 100 / self.scale, fovx, fovy).to(self.device).T.contiguous()
            cam = self.MiniCam(width, height, fovy, fovx, .01 / self.scale, 100 / self.scale, view, view @ projection)
            result = self.render_function(cam, self.model, self.pipeline, self.background)
            rgb = result["render"].clamp(0, 1).permute(1, 2, 0)
            alpha = result["rend_alpha"].clamp(0, 1).permute(1, 2, 0)
            rgba = torch.cat([rgb, alpha], dim=2).mul(255).byte().cpu().numpy()
            depth = (result["surf_depth"][0] * self.scale).nan_to_num(0).float().cpu().numpy()
        return rgba, depth


class GSplat3DGSRenderer(NativeCoordinateRenderer):
    """Adapter for standard Graphdeco 3DGS PLY; rejects 2D scale schemas."""
    def __init__(self, cfg, alignment, device):
        super().__init__(cfg, alignment, device)
        from plyfile import PlyData
        from gsplat import rasterization
        torch = self.torch
        ply = PlyData.read(str(cfg.asset(cfg.data["gaussian"]["path"])))["vertex"]
        names = ply.data.dtype.names
        scales = sorted((n for n in names if n.startswith("scale_")), key=lambda n: int(n.split("_")[-1]))
        if len(scales) != 3:
            raise ValueError("gsplat_3dgs expects three scale fields; use matcha_2dgs for MAtCha")
        tensor = lambda array: torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float32, device=device)
        self.means = tensor(np.column_stack([ply[k] for k in ("x", "y", "z")]))
        self.scales = tensor(np.column_stack([ply[k] for k in scales])).exp()
        self.quats = torch.nn.functional.normalize(tensor(np.column_stack([ply[f"rot_{i}"] for i in range(4)])), dim=-1)
        self.opacity = tensor(ply["opacity"]).sigmoid()
        dc = np.column_stack([ply[f"f_dc_{i}"] for i in range(3)])[:, None, :]
        rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1]))
        sh = dc if not rest else np.concatenate([dc, np.column_stack([ply[k] for k in rest]).reshape(len(ply), 3, -1).transpose(0, 2, 1)], axis=1)
        self.sh_degree = round(math.sqrt(sh.shape[1]) - 1)
        if (self.sh_degree + 1) ** 2 != sh.shape[1]:
            raise ValueError("Invalid spherical harmonic coefficient count")
        self.colors, self.rasterization, self.count = tensor(sh), rasterization, len(ply)

    def render(self, camera, intrinsics, width, height):
        torch = self.torch
        view = torch.as_tensor(np.linalg.inv(self.native_camera(camera))[None].copy(), dtype=torch.float32, device=self.device)
        k = torch.as_tensor(np.asarray(intrinsics)[None], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            rgbd, alpha, _ = self.rasterization(self.means, self.quats, self.scales, self.opacity,
                self.colors, view, k, width, height, sh_degree=self.sh_degree,
                near_plane=.01 / self.scale, far_plane=100 / self.scale, packed=True, render_mode="RGB+ED")
            rgba = torch.cat([rgbd[0, ..., :3].clamp(0, 1), alpha[0]], dim=-1).mul(255).byte().cpu().numpy()
            depth = (rgbd[0, ..., 3] * self.scale).float().cpu().numpy()
        return rgba, depth


register_backend("matcha_2dgs", Matcha2DGSRenderer)
register_backend("gsplat_3dgs", GSplat3DGSRenderer)
