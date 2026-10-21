"""Isolated CUDA renderer with a bounded local HTTP protocol."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from .config import SceneConfig
from .backends import create_backend


def serve(config, port, token, device):
    cfg = SceneConfig.read(config)
    alignment = json.loads((cfg.cache / "alignment.json").read_text())
    renderer = create_backend(cfg, alignment, device)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data = json.dumps({"ready": True, "token": token, "gaussians": renderer.count,
                               "backend": cfg.data["gaussian"]["backend"]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/render":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 1 <= length <= 10000:
                    raise ValueError("Invalid request length")
                data = json.loads(self.rfile.read(length))
                if data.get("token") != token:
                    self.send_error(403)
                    return
                width, height = int(data["width"]), int(data["height"])
                camera, k = np.asarray(data["camera"]), np.asarray(data["K"])
                if not 64 <= width <= 1280 or not 64 <= height <= 960:
                    raise ValueError("Unsupported resolution")
                if camera.shape != (4, 4) or k.shape != (3, 3) or not np.isfinite(camera).all() or not np.isfinite(k).all():
                    raise ValueError("Invalid camera")
                rgba, depth = renderer.render(camera, k, width, height)
                payload = np.asarray(rgba, dtype=np.uint8).tobytes() + np.asarray(depth, dtype="<f4").tobytes()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-State-ID", str(int(data["state_id"])))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                self.send_error(500, str(error))

        def log_message(self, *args):
            pass

    print("GAUSSIAN_WORKER_READY", renderer.count, flush=True)
    try:
        HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    finally:
        renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    serve(args.scene, args.port, args.token, args.device)
