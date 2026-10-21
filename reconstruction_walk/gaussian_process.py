"""Lifecycle and frame-checked communication with the Gaussian CUDA worker."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import uuid

import numpy as np


class GaussianProcess:
    def __init__(self, cfg, device="cuda:0", timeout=180):
        from .config import REPO
        self.token = uuid.uuid4().hex
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log_path = cfg.cache / "gaussian-worker.log"
        self.log = self.log_path.open("w")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "reconstruction_walk.worker",
            "--scene", str(cfg.path), "--port", str(port), "--token", self.token, "--device", device],
            cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        started = time.monotonic()
        last_notice = started
        try:
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Gaussian worker failed:\n{self.log_path.read_text()[-6000:]}")
                try:
                    with urllib.request.urlopen(self.url, timeout=.5) as response:
                        state = json.load(response)
                    if state.get("ready") and state.get("token") == self.token:
                        print("[GS] Ready:", state["backend"], state["gaussians"], flush=True)
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"Gaussian worker startup timeout; see {self.log_path}")
                if time.monotonic() - last_notice > 15:
                    print("[GS] Loading/initializing CUDA; log:", self.log_path, flush=True)
                    last_notice = time.monotonic()
                time.sleep(.2)
        except BaseException:
            self.close()
            raise

    def render(self, state_id, camera, intrinsics, width, height):
        body = json.dumps({"token": self.token, "state_id": int(state_id), "camera": camera.tolist(),
                           "K": intrinsics, "width": width, "height": height}).encode()
        request = urllib.request.Request(self.url + "/render", data=body,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if int(response.headers["X-State-ID"]) != state_id:
                raise RuntimeError("Gaussian response state ID does not match snapshot")
            raw = response.read()
        n = width * height
        if len(raw) != n * 8:
            raise RuntimeError("Unexpected Gaussian response size")
        return (np.frombuffer(raw[:n * 4], np.uint8).reshape(height, width, 4).copy(),
                np.frombuffer(raw[n * 4:], "<f4").reshape(height, width).copy())

    def close(self):
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if getattr(self, "log", None) is not None:
            self.log.close()
