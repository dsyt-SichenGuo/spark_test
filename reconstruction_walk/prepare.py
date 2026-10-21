"""Prepare independent metric visual/collision USDs and reusable spawn positions."""

import json
from pathlib import Path
import time

import numpy as np
from plyfile import PlyData

from .config import SceneConfig
from .geometry import calibrate, cluster_mesh, similarity, transform_points, SurfaceQueries


def write_visual(path, vertices, faces, colors, scale, rotation, translation):
    from pxr import Usd, UsdGeom, UsdShade, Vt, Sdf, Gf
    temporary = path.with_name(path.stem + ".building.usdc")
    if temporary.exists():
        temporary.unlink()
    stage = Usd.Stage.CreateNew(str(temporary))
    root = UsdGeom.Xform.Define(stage, "/Visual")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = scale * rotation, translation
    root.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
    mesh = UsdGeom.Mesh.Define(stage, "/Visual/Surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorPrimvar("vertex").Set(Vt.Vec3fArray.FromNumpy(colors))
    material = UsdShade.Material.Define(stage, "/Visual/Material")
    shader = UsdShade.Shader.Define(stage, "/Visual/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, "/Visual/Material/Color")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    del stage
    temporary.replace(path)


def write_collision(path, vertices, faces):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Vt
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/Collision")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    mesh = UsdGeom.Mesh.Define(stage, "/Collision/Surface")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(vertices, dtype=np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(faces.reshape(-1), dtype=np.int32)))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    material = UsdShade.Material.Define(stage, "/Collision/Friction")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.9)
    physics.CreateDynamicFrictionAttr(.8)
    physics.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()


def prepare(config_path, force=False):
    cfg = SceneConfig.read(config_path)
    cache = cfg.cache
    cache.mkdir(parents=True, exist_ok=True)
    signature = cfg.asset_signature()
    report_path = cache / "prepared.json"
    expected = ["visual.usdc", "collision.usdc", "collision.npz", "alignment.json", "spawns.npy"]
    if not force and report_path.exists() and all((cache / name).exists() for name in expected):
        previous = json.loads(report_path.read_text())
        if previous.get("signature") == signature and previous.get("version") == 1:
            print("[prepare] Reusing current metric scene cache:", cache, flush=True)
            return cfg, previous
    start = time.monotonic()
    mesh_path = cfg.asset(cfg.data["mesh"]["path"])
    print("[prepare] Reading colored triangle mesh:", mesh_path, flush=True)
    ply = PlyData.read(str(mesh_path), known_list_len={"face": {"vertex_indices": 3}})
    vertices = np.column_stack([ply["vertex"][k] for k in ("x", "y", "z")]).astype(np.float32)
    faces = np.asarray(ply["face"]["vertex_indices"], dtype=np.int32)
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Invalid mesh coordinates or topology")
    cameras = json.loads(cfg.asset(cfg.data["cameras"]).read_text())
    centers = np.array([c["position"] for c in cameras])
    rotations = np.array([c["rotation"] for c in cameras])
    # Optional Gaussian-native -> mesh-native transform for methods exporting
    # separately aligned appearance and surfaces.
    gs, gr, gt = similarity(cfg.data["gaussian"].get("to_mesh"))
    centers_mesh = transform_points(centers, gs, gr, gt)
    rotations_mesh = gr @ rotations
    print("[prepare] Fitting floor and metric coordinate transform", flush=True)
    scale, rotation, translation, calibration = calibrate(
        vertices, faces, centers_mesh, rotations_mesh,
        cfg.data["camera_height_m"], cfg.data.get("meters_per_unit"))
    world = transform_points(vertices, scale, rotation, translation).astype(np.float32)
    camera_world = transform_points(centers_mesh, scale, rotation, translation)
    camera_rot_world = rotation @ rotations_mesh
    voxel = float(cfg.data.get("collision_voxel_m", .06))
    for _ in range(5):
        print(f"[prepare] Clustering collision mesh at {voxel:.3f} m", flush=True)
        collision_vertices, collision_faces = cluster_mesh(world, faces, voxel)
        print(f"[prepare] Collision: {len(collision_vertices):,} vertices, {len(collision_faces):,} triangles", flush=True)
        if len(collision_faces) <= int(cfg.data.get("max_collision_faces", 800000)):
            break
        voxel *= 1.35
    else:
        raise ValueError("Collision budget exceeded; increase collision_voxel_m or max_collision_faces")
    print("[prepare] Building indexed surface queries and validating spawn footprints", flush=True)
    query = SurfaceQueries(collision_vertices, collision_faces)
    spawns = query.spawn_candidates(camera_world, camera_rot_world,
        cfg.data.get("footprint_radius_m", .28), cfg.data.get("spawn_clearance_m", .45))
    print(f"[prepare] Found {len(spawns)} supported, clear spawn positions", flush=True)
    np.savez(cache / "collision.npz", vertices=collision_vertices, faces=collision_faces)
    np.save(cache / "spawns.npy", spawns)
    alignment = {"scale": scale * gs, "rotation": (rotation @ gr).tolist(),
                 "translation": (scale * rotation @ gt + translation).tolist()}
    (cache / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    print("[prepare] Writing collision USD and complete colored visual USD", flush=True)
    write_collision(cache / "collision.usdc", collision_vertices, collision_faces)
    if {"red", "green", "blue"}.issubset(ply["vertex"].data.dtype.names):
        rgb = np.column_stack([ply["vertex"][k] for k in ("red", "green", "blue")]).astype(np.float32) / 255
    else:
        rgb = np.full((len(vertices), 3), .65, dtype=np.float32)
    linear = np.ascontiguousarray(np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4), dtype=np.float32)
    write_visual(cache / "visual.usdc", vertices, faces, linear, scale, rotation, translation)
    report = {"version": 1, "signature": signature, "calibration": calibration,
              "mesh_to_world": {"scale": scale, "rotation": rotation.tolist(), "translation": translation.tolist()},
              "gaussian_to_world": alignment, "visual_vertices": len(vertices), "visual_faces": len(faces),
              "collision_vertices": len(collision_vertices), "collision_faces": len(collision_faces),
              "collision_voxel_m": voxel, "spawn_count": len(spawns), "initial_spawn_xy_floor_yaw": spawns[0].tolist(),
              "camera_centers_world": camera_world.tolist(), "bounds": [world.min(0).tolist(), world.max(0).tolist()],
              "elapsed_seconds": time.monotonic() - start}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("[prepare] Completed:", json.dumps({k: report[k] for k in (
        "calibration", "collision_faces", "spawn_count", "initial_spawn_xy_floor_yaw", "elapsed_seconds")}), flush=True)
    return cfg, report


if __name__ == "__main__":
    import argparse
    from .config import DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.scene, args.force)
