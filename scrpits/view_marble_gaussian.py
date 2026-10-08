"""Browser preview of the original Marble SPZ appearance, without Isaac or people.

Requires NumPy and viser >= 1.1.1. Run from any directory; defaults to the
bundled 500k-point asset. Use --quality full for all 1.92 million Gaussians.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import struct
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = REPO_ROOT / "reconstructions/marble-1.1-supermarket-full-appearance-1s-18s/assets/marble_1_1_people_1s_18s"
ASSET_NAMES = {"100k": "world_100k.spz", "150k": "world_150k.spz",
               "500k": "world_500k.spz", "full": "world_full_res.spz"}


def load_spz(path):
    """Decode the package's SPZ v2, degree-zero Gaussians, in source coordinates."""
    with gzip.open(path, "rb") as stream:
        data = stream.read()
    if len(data) < 16:
        raise ValueError("Truncated SPZ header")
    magic, version, count, degree, bits, flags, _ = struct.unpack_from("<III4B", data)
    if (magic, version, degree, flags) != (0x5053474E, 2, 0, 0):
        raise ValueError("This preview supports the bundled SPZ v2 degree-zero assets only")
    if count == 0 or len(data) != 16 + count * 19:
        raise ValueError("Invalid SPZ payload length")
    offset = 16

    def take(size):
        nonlocal offset
        values = np.frombuffer(data, np.uint8, size, offset)
        offset += size
        return values

    packed = take(count * 9).reshape(count, 3, 3).astype(np.int32)
    xyz = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16)
    xyz = np.where(xyz & 0x800000, xyz - 0x1000000, xyz)
    centers = xyz.astype(np.float32) / (1 << bits)
    opacities = take(count).astype(np.float32).reshape(count, 1) / 255
    dc = (take(count * 3).reshape(count, 3).astype(np.float32) / 255 - .5) / .15
    colors = np.clip(dc * .28209479177387814 + .5, 0, 1)
    scales = np.exp(take(count * 3).reshape(count, 3).astype(np.float32) / 16 - 10)
    qxyz = take(count * 3).reshape(count, 3).astype(np.float32) / 127.5 - 1
    qw = np.sqrt(np.maximum(0, 1 - np.sum(qxyz * qxyz, axis=1)))
    norm = np.sqrt(qw * qw + np.sum(qxyz * qxyz, axis=1))
    x, y, z = (qxyz / norm[:, None]).T
    w = qw / norm
    rotation = np.empty((count, 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y*y + z*z)
    rotation[:, 0, 1] = 2 * (x*y - z*w)
    rotation[:, 0, 2] = 2 * (x*z + y*w)
    rotation[:, 1, 0] = 2 * (x*y + z*w)
    rotation[:, 1, 1] = 1 - 2 * (x*x + z*z)
    rotation[:, 1, 2] = 2 * (y*z - x*w)
    rotation[:, 2, 0] = 2 * (x*z - y*w)
    rotation[:, 2, 1] = 2 * (y*z + x*w)
    rotation[:, 2, 2] = 1 - 2 * (x*x + y*y)
    scaled = rotation * scales[:, None, :]
    covariances = scaled @ scaled.transpose(0, 2, 1)
    return centers, covariances, colors, opacities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", choices=ASSET_NAMES, default="500k")
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--port", type=int, default=8878)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--check-only", action="store_true", help="decode and validate without starting a server")
    args = parser.parse_args()
    asset = args.asset_dir / ASSET_NAMES[args.quality]
    centers, covariances, colors, opacities = load_spz(asset)
    for values in (centers, covariances, colors, opacities):
        if not np.isfinite(values).all():
            raise ValueError("Non-finite Gaussian data")
    print(f"Loaded {len(centers):,} Gaussians from {asset}", flush=True)
    if args.check_only:
        return
    import viser
    if not hasattr(viser, "SceneApi") or not hasattr(viser.SceneApi, "add_gaussian_splats"):
        parser.error("Install a recent viser in a separate environment: python -m pip install 'viser>=1.1.1'")
    server = viser.ViserServer(host=args.host, port=args.port, label="Marble supermarket")
    server.scene.set_up_direction("+z")
    server.scene.world_axes.visible = False
    # Match the original renderer's source RUB -> USD Z-up rotation (-90 deg X).
    server.scene.add_gaussian_splats("/Marble", centers, covariances, colors, opacities,
                                    wxyz=(2**-.5, -2**-.5, 0., 0.))
    server.gui.add_markdown(f"**Marble original appearance** · {len(centers):,} Gaussians\n\n"
                            "Drag to look/orbit; right-drag to pan; scroll to zoom. "
                            "Use the viewer camera controls for navigation.\n\n"
                            "Static SPZ only: no robot or animated people are loaded.")
    reset = server.gui.add_button("Reset indoor view")

    def reset_camera(client):
        with client.atomic():
            client.camera.up_direction = (0., 0., 1.)
            client.camera.position = (0., 0., 0.)
            client.camera.look_at = (-1., -6., 0.)
            client.camera.fov = np.deg2rad(65.)

    @server.on_client_connect
    def connected(client):
        reset_camera(client)

    @reset.on_click
    def reset_clicked(event):
        if event.client is not None:
            reset_camera(event.client)

    print(f"Open http://{args.host}:{server.get_port()} (Ctrl+C to stop)", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
