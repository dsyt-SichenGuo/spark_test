"""Metric alignment, geometry-preserving vertex clustering and indexed queries."""

import math
import numpy as np


def similarity(value=None):
    value = value or {}
    r = np.asarray(value.get("rotation", np.eye(3)), dtype=float)
    t = np.asarray(value.get("translation", [0, 0, 0]), dtype=float)
    s = float(value.get("scale", 1.))
    if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError("Invalid similarity transform dimensions/values")
    if s <= 0 or not math.isfinite(s) or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Similarity requires positive scale and a proper rotation")
    return s, r, t


def transform_points(points, s, r, t):
    return np.asarray(points) @ r.T * s + t


def calibrate(vertices, faces, camera_centers, camera_rotations, camera_height=1.55, scale=None):
    """Find a dominant floor near the recording path, oriented using camera up."""
    step = max(1, len(faces) // 150000)
    triangles = vertices[faces[::step]]
    centers = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], 1e-12)
    hint = -np.mean(camera_rotations[:min(10, len(camera_rotations)), :, 1], axis=0)
    hint /= np.linalg.norm(hint)
    keep = (np.abs(normals @ hint) > .8) & (norms > 1e-10)
    centers, normals = centers[keep], normals[keep]
    if len(centers) < 100:
        raise ValueError("Not enough floor-like mesh triangles to calibrate")
    normals *= np.where(normals @ hint < 0, -1., 1.)[:, None]
    rng = np.random.default_rng(17)
    best = None
    for i in rng.choice(len(normals), min(180, len(normals)), replace=False):
        normal, d = normals[i], float(centers[i] @ normals[i])
        heights = camera_centers @ normal - d
        if np.percentile(heights, 5) < .08 or not .15 < np.median(heights) < 5:
            continue
        inliers = np.abs(centers @ normal - d) < .035
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, normal.copy(), d)
    if best is None:
        raise ValueError("Cannot identify supported floor below the recording cameras")
    _, normal, d = best
    for _ in range(3):
        points = centers[np.abs(centers @ normal - d) < .05]
        mean = points.mean(axis=0)
        _, _, vt = np.linalg.svd(points - mean, full_matrices=False)
        normal = vt[-1]
        if normal @ hint < 0:
            normal *= -1
        d = float(mean @ normal)
    height_native = float(np.median(camera_centers @ normal - d))
    metric_scale = float(scale if scale is not None else camera_height / height_native)
    forward = camera_rotations[0, :, 2].copy()
    forward -= normal * (forward @ normal)
    forward /= np.linalg.norm(forward)
    left = np.cross(normal, forward)
    rotation = np.stack([forward, left, normal])
    translation = -metric_scale * rotation @ camera_centers[0]
    translation[2] = -metric_scale * d
    return metric_scale, rotation, translation, {
        "floor_normal_native": normal.tolist(), "floor_offset_native": d,
        "median_camera_height_native": height_native,
        "scale_source": "explicit meters_per_unit" if scale is not None else "assumed recording camera height",
        "assumed_camera_height_m": camera_height if scale is None else None,
        "meters_per_native_unit": metric_scale,
        "floor_fit_inliers": int(np.sum(np.abs(centers @ normal - d) < .05)),
    }


def cluster_mesh(vertices, faces, voxel):
    """Merge nearby vertices; retain surviving surface triangles, never random faces."""
    cells = np.floor(vertices / voxel).astype(np.int32)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.astype(np.int32, copy=False)
    count = np.bincount(inverse)
    points = np.column_stack([np.bincount(inverse, weights=vertices[:, a]) / count for a in range(3)]).astype(np.float32)
    chunks = []
    for start in range(0, len(faces), 1000000):
        f = inverse[faces[start:start + 1000000]]
        valid = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        chunks.append(f[valid])
    triangles = np.concatenate(chunks)
    _, unique = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[unique]
    cross = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                     points[triangles[:, 2]] - points[triangles[:, 0]])
    triangles = triangles[np.linalg.norm(cross, axis=1) > 1e-8]
    return points, triangles


class SurfaceQueries:
    def __init__(self, vertices, faces):
        import trimesh
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.ray = trimesh.ray.ray_triangle.RayMeshIntersector(self.mesh)

    def heights(self, xy, reference_z=0., tolerance=.55):
        xy = np.asarray(xy).reshape(-1, 2)
        origins = np.column_stack([xy, np.full(len(xy), reference_z + 1.)])
        directions = np.tile([0., 0., -1.], (len(xy), 1))
        locations, rays, _ = self.ray.intersects_location(origins, directions, multiple_hits=True)
        heights = np.full(len(xy), -np.inf)
        if len(locations):
            valid = np.abs(locations[:, 2] - reference_z) <= tolerance
            np.maximum.at(heights, rays[valid], locations[valid, 2])
        heights[~np.isfinite(heights)] = np.nan
        return heights

    def camera_eye(self, target, eye):
        target, eye = np.asarray(target), np.asarray(eye)
        direction = eye - target
        length = np.linalg.norm(direction)
        if length < 1e-5:
            return eye
        locations, _, _ = self.ray.intersects_location(target[None], (direction / length)[None])
        if not len(locations):
            return eye
        distances = np.linalg.norm(locations - target, axis=1)
        distances = distances[(distances > .02) & (distances < length)]
        if len(distances):
            return target + direction * max(.05, (distances.min() - .15) / length)
        return eye

    def spawn_candidates(self, camera_centers, camera_rotations, radius=.28, clearance=.45):
        offsets = np.array([(x, y) for x in (-.4, 0., .4) for y in (-.4, 0., .4)])
        xy = (camera_centers[:, None, :2] + offsets[None]).reshape(-1, 2)
        xy = np.unique(np.round(xy, 2), axis=0)
        footprint = np.array([(x, y) for x in (-radius, 0, radius) for y in (-radius, 0, radius)])
        height = self.heights((xy[:, None] + footprint[None]).reshape(-1, 2)).reshape(len(xy), -1)
        valid = np.all(np.isfinite(height), axis=1)
        valid &= np.ptp(np.nan_to_num(height, nan=99), axis=1) < .12
        xy, floor = xy[valid], height[valid, 4]
        candidates = []
        for point, z in zip(xy, floor):
            angles = np.arange(8) * np.pi / 4
            horizontal = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(8)])
            origins = np.repeat(np.array([[point[0], point[1], z + .2],
                                         [point[0], point[1], z + .8],
                                         [point[0], point[1], z + 1.3]]), 8, axis=0)
            directions = np.tile(horizontal, (3, 1))
            locations, rays, _ = self.ray.intersects_location(origins, directions)
            distance = np.linalg.norm(locations - origins[rays], axis=1)
            nearest = float(distance.min()) if len(distance) else 10.
            if nearest < clearance:
                continue
            origin = np.array([[point[0], point[1], z + .05]])
            roof, _, _ = self.ray.intersects_location(origin, np.array([[0., 0., 1.]]))
            if len(roof) and np.min(roof[:, 2] - z) < 1.7:
                continue
            near = int(np.argmin(np.linalg.norm(camera_centers[:, :2] - point, axis=1)))
            forward = camera_rotations[near, :, 2]
            yaw = math.atan2(forward[1], forward[0])
            score = .2 * min(nearest, 3) - np.linalg.norm(point - camera_centers[0, :2])
            candidates.append((score, [float(point[0]), float(point[1]), float(z), yaw]))
        if not candidates:
            raise ValueError("No clear supported spawn found; inspect calibration or provide another mesh")
        candidates.sort(key=lambda item: item[0], reverse=True)
        return np.array([item[1] for item in candidates])
