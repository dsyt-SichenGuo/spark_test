"""Two windows publish the exact same captured simulation state and camera."""

from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np


def robot_pixel_mask(instances, shape):
    if not isinstance(instances, dict):
        raise ValueError("Isaac instance labels are unavailable")
    ids = np.asarray(instances.get("data", [])).reshape(shape)
    labels = instances.get("info", {}).get("idToLabels", {})
    robot_ids = []
    for key, label in labels.items():
        path = label.get("primPath", "") if isinstance(label, dict) else label
        if "/Robot" in str(path) and not str(path).startswith("/World/WalkEnvironment"):
            robot_ids.append(int(key))
    return np.isin(ids, robot_ids)


def composite(gs_rgba, gs_depth, native_rgba, native_depth, robot_mask):
    z = np.asarray(native_depth).reshape(gs_depth.shape)
    native_valid = robot_mask & np.isfinite(z) & (z > .01)
    gs_valid = (gs_rgba[:, :, 3] > 64) & np.isfinite(gs_depth) & (gs_depth > .01)
    visible = native_valid & (~gs_valid | (z <= gs_depth + .025))
    image = gs_rgba.copy()
    image[visible, :3] = native_rgba[visible, :3]
    image[:, :, 3] = 255
    return image


class DualViewRenderer:
    def __init__(self, agent, cfg, query, worker, width=640, height=480, light_scale=2., mesh_appearance=None,
                 video_dir=None, video_fps=30., record=False):
        import omni.replicator.core as rep
        import omni.ui as ui
        import omni.usd
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import UsdGeom, UsdLux, Usd, Gf, Sdf
        import carb
        self.agent, self.query, self.worker, self.cfg = agent, query, worker, cfg
        self.mesh_appearance = mesh_appearance
        from .recording import PairedVideoRecorder
        self.recorder = PairedVideoRecorder(video_dir or cfg.cache / "videos", video_fps) if worker is not None else None
        self.record_requested = record
        self.width, self.height = width, height
        self.stage = omni.usd.get_context().get_stage()
        settings = carb.settings.get_settings()
        settings.set("/rtx/useViewLightingMode", False)
        settings.set("/rtx/sceneDb/ambientLightIntensity", .4 * light_scale)
        self.Gf = Gf
        self.camera = UsdGeom.Camera.Define(self.stage, "/World/WalkCamera")
        self.transform = self.camera.AddTransformOp()
        self.camera.CreateClippingRangeAttr((.01, 100.))
        import json
        cameras = json.loads(cfg.asset(cfg.data["cameras"]).read_text())
        c = cameras[0]
        focal = float(c["fx"]) / c["width"] * width
        self.k = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
        self.camera.CreateFocalLengthAttr(18.)
        self.camera.CreateHorizontalApertureAttr(width * 18 / focal)
        self.camera.CreateVerticalApertureAttr(height * 18 / focal)
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            dome = UsdLux.DomeLight.Define(self.stage, "/World/WalkLights/Dome")
            dome.CreateIntensityAttr(1000 * light_scale)
            UsdLux.ShadowAPI.Apply(dome.GetPrim()).CreateShadowEnableAttr(False)
            for name, pos, intensity in (("Key", (-.35, .3, .15), 1800), ("Fill", (.35, -.15, .15), 900)):
                light = UsdLux.RectLight.Define(self.stage, f"/World/WalkCamera/{name}")
                light.CreateIntensityAttr(intensity * light_scale)
                light.CreateWidthAttr(1.5)
                light.CreateHeightAttr(1.5)
                light.CreateNormalizeAttr(True)
                UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)
                UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*pos))
            # A dome alone cannot illuminate a closed interior well. Distribute
            # shadow-free mesh-only fill along the known reconstruction path.
            report = json.loads((cfg.cache / "prepared.json").read_text())
            last = None
            for i, position in enumerate(report["camera_centers_world"]):
                point = np.asarray(position)
                if last is not None and np.linalg.norm(point - last) < 3.:
                    continue
                last = point
                light = UsdLux.SphereLight.Define(self.stage, f"/World/WalkLights/Interior_{i}")
                light.CreateIntensityAttr(1600 * light_scale)
                light.CreateRadiusAttr(.25)
                light.CreateNormalizeAttr(True)
                UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(False)
                UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*map(float, point)))
                link = UsdLux.LightAPI(light.GetPrim()).GetLightLinkCollectionAPI()
                link.CreateIncludeRootAttr(False)
                link.CreateExpansionRuleAttr(Usd.Tokens.expandPrims)
                link.CreateIncludesRel().SetTargets([Sdf.Path("/World/WalkEnvironment/Visual")])
        self.product = rep.create.render_product(str(self.camera.GetPath()), (width, height))
        self.rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self.ids = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        for annotator in (self.rgb, self.depth, self.ids):
            annotator.attach([self.product])
        self.rep = rep
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="GaussianRequest")
        self.pending = None
        self.eye = None
        self.last_yaw = 0.
        self.last_sequence = None
        self.saved = False
        self.mesh_provider = ui.ByteImageProvider()
        self.mesh_window = ui.Window("G1 | Mesh environment", width=width + 20, height=height + 80)
        with self.mesh_window.frame:
            with ui.VStack():
                if mesh_appearance is not None:
                    with ui.HStack(height=24):
                        self.mesh_color_model = ui.SimpleBoolModel(mesh_appearance.colored)
                        ui.CheckBox(model=self.mesh_color_model, width=24)
                        ui.Label("Mesh source colors", tooltip="Off: neutral gray geometry. On: original reconstructed colors.")
                        self.mesh_color_model.add_value_changed_fn(self._mesh_color_changed)
                self.mesh_status = ui.Label("Preparing mesh + G1", height=25)
                ui.ImageWithProvider(self.mesh_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                ui.Label("W/S forward/back | A/D turn | Q/E strafe | Space stop | R respawn", height=25)
        self.gs_window = None
        if worker is not None:
            self.gs_provider = ui.ByteImageProvider()
            self.gs_window = ui.Window("G1 | Gaussian environment", width=width + 20, height=height + 80)
            with self.gs_window.frame:
                with ui.VStack():
                    with ui.HStack(height=25):
                        self.record_button = ui.Button("Start recording (F9)", width=175,
                                                       clicked_fn=self.toggle_recording)
                        self.video_status = ui.Label("Recording off", tooltip=str(self.recorder.directory))
                    self.gs_status = ui.Label("Preparing GS + same G1", height=25)
                    ui.ImageWithProvider(self.gs_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    ui.Label("Same state, camera and metric depth as the mesh window", height=25)
        viewport = get_active_viewport()
        if viewport:
            viewport.camera_path = str(self.camera.GetPath())
            viewport.resolution = (width, height)
        target = ui.Workspace.get_window("Viewport")
        if target:
            self.mesh_window.dock_in(target, ui.DockPosition.SAME)
            if self.gs_window:
                self.gs_window.dock_in(self.mesh_window, ui.DockPosition.RIGHT, .5)

    def toggle_recording(self):
        if self.recorder is None:
            print("[video] Recording requires both GS and mesh views", flush=True)
            return
        try:
            if self.recorder.active or self.recorder.error is not None:
                path = self.recorder.stop()
                self.record_requested = False
                self.video_status.text = f"Saved: {path.name}" if path else "Recording off"
                if path:
                    self.video_status.tooltip = str(path)
            elif self.record_requested:
                self.record_requested = False
                self.video_status.text = "Recording off"
            elif self.recorder.ready:
                self.recorder.start()
            else:
                self.record_requested = True
            self._recording_status()
        except Exception as exc:
            self.record_requested = False
            self.record_button.text = "Start recording (F9)"
            self.video_status.text = f"Recording error: {exc}"
            print(f"[video] {exc}", flush=True)

    def _recording_status(self):
        if self.recorder is None:
            return
        if self.recorder.error is not None:
            self.video_status.text = f"Recording error: {self.recorder.error}"
            self.record_button.text = "Stop recording (F9)"
        elif self.recorder.active:
            self.video_status.text = f"REC {self.recorder.elapsed:.1f}s | {self.recorder.fps:g} FPS | GS left / mesh right"
            self.video_status.tooltip = str(self.recorder.path)
            self.record_button.text = "Stop recording (F9)"
        elif self.record_requested:
            self.video_status.text = "Waiting for the first synchronized frame"
            self.record_button.text = "Cancel recording (F9)"
        else:
            self.record_button.text = "Start recording (F9)"

    def _mesh_color_changed(self, model):
        self.mesh_appearance.set_colored(model.as_bool)
        # Discard the pending image with the previous material. Both windows
        # will publish together again on the next captured simulation state.
        if self.pending is not None:
            self.pending[1]["discard"] = True

    def camera_pose(self, root):
        x, y, z, qx, qy, qz, qw = root
        up = 1 - 2 * (qx * qx + qy * qy)
        if up > .7:
            self.last_yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        forward = np.array([math.cos(self.last_yaw), math.sin(self.last_yaw), 0.])
        target = np.array([x, y, z + .3])
        desired = np.array([x, y, z + 1.0]) - 2 * forward
        if self.eye is None:
            self.eye = desired
        self.eye += .35 * (desired - self.eye)
        self.eye = self.query.camera_eye(target, self.eye)
        matrix = self.Gf.Matrix4d().SetLookAt(self.Gf.Vec3d(*map(float, self.eye)),
                    self.Gf.Vec3d(*map(float, target)), self.Gf.Vec3d(0, 0, 1)).GetInverse()
        self.transform.Set(matrix)
        # USD -Z forward / +Y up -> CV +Z forward / +Y down.
        return np.asarray(matrix).T @ np.diag([1., -1., -1., 1.])

    def invalidate(self):
        # Drop the old visual result on reset; it must never appear beside a
        # post-reset frame. The running request is allowed to finish safely.
        if self.pending is not None:
            self.pending[1]["discard"] = True
        self.eye = None

    def publish_ready(self):
        self._recording_status()
        if self.pending is None or not self.pending[0].done():
            return
        future, snapshot = self.pending
        self.pending = None
        rgba, depth = future.result()
        if snapshot.get("discard"):
            return
        image = composite(rgba, depth, snapshot["rgb"], snapshot["depth"], snapshot["mask"])
        self._publish(snapshot, image)

    def _publish(self, snapshot, gs_image=None):
        self.mesh_provider.set_data_array(snapshot["rgb"], [self.width, self.height])
        self.mesh_status.text = f"Mesh + G1 | state {snapshot['sequence']} | {snapshot['policy']} | {snapshot['time']:.2f}s"
        if gs_image is not None:
            self.gs_provider.set_data_array(gs_image, [self.width, self.height])
            self.gs_status.text = f"GS + G1 | same state {snapshot['sequence']} | depth occlusion enabled"
            self.recorder.update(gs_image, snapshot["rgb"])
            if self.record_requested:
                # The request can precede initialization; encoding starts only
                # once a complete pair has actually been published.
                self.record_requested = False
                self.toggle_recording()
        self.last_sequence = snapshot["sequence"]
        if not self.saved and snapshot["sequence"] >= 70:
            from PIL import Image
            Image.fromarray(snapshot["rgb"]).save(self.cfg.cache / "walk-mesh.png")
            Image.fromarray(snapshot["rgb"]).save(self.cfg.cache / f"walk-mesh-{snapshot['policy']}.png")
            if gs_image is not None:
                Image.fromarray(gs_image).save(self.cfg.cache / "walk-gaussian.png")
                Image.fromarray(gs_image).save(self.cfg.cache / f"walk-gaussian-{snapshot['policy']}.png")
            self.saved = True

    def capture(self, sequence, policy, sim_time, root):
        self.publish_ready()
        if self.pending is not None:
            return
        camera = self.camera_pose(root)
        state = self.agent._snapshot_camera_render_state()
        try:
            # Replicator capture is atomic w.r.t. the G1 physical state.
            self.rep.orchestrator.step(rt_subframes=2, delta_time=0., pause_timeline=False)
            rgb = np.ascontiguousarray(self.rgb.get_data()).copy()
            depth = np.asarray(self.depth.get_data()).copy()
            instances = self.ids.get_data()
        finally:
            self.agent._restore_camera_render_state(state)
        if rgb.shape != (self.height, self.width, 4) or depth.size != self.width * self.height:
            return
        snapshot = {"rgb": rgb, "depth": depth.reshape(self.height, self.width),
                    "sequence": sequence, "policy": policy, "time": sim_time}
        if self.worker is None:
            self._publish(snapshot)
            return
        snapshot["mask"] = robot_pixel_mask(instances, (self.height, self.width))
        if not snapshot["mask"].any():
            self.mesh_status.text = "Waiting for G1 instance labels / camera visibility"
        future = self.executor.submit(self.worker.render, sequence, camera, self.k, self.width, self.height)
        self.pending = future, snapshot

    def close(self):
        if self.recorder is not None:
            try:
                self.recorder.stop()
            except Exception as exc:
                print(f"[video] Error finalizing recording: {exc}", flush=True)
        self.executor.shutdown(wait=True, cancel_futures=True)
        for annotator in (self.rgb, self.depth, self.ids):
            annotator.detach([self.product])
        self.product.destroy()
        self.mesh_window.destroy()
        if self.gs_window:
            self.gs_window.destroy()
