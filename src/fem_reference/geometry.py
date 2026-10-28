"""Independent geometry reconstruction for R2-THERMAL-FEM-02.

The reference labels store 5000 points per geometry (the first 660 are boundary /
near-wall points).  This module reconstructs the exact outer square and the
centred regular-polygon hole from the verified geometry specification and checks
that reconstruction against the original boundary points.  It then builds an
independent polar triangle mesh of the annular domain; no gmsh/FEniCS is needed.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

GEOMETRY_SPEC_PATH = os.path.join(WORKSPACE, "results",
                                  "thermal_geometry_spec.json")
SPLIT_CONFIG_PATH = os.path.join(WORKSPACE, "configs", "splits_thermal.json")
BOUNDARY_N = 660
OUTER_N = 492
INNER_N = BOUNDARY_N - OUTER_N


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_geometry_spec(path: str = GEOMETRY_SPEC_PATH) -> dict:
    doc = load_json(path)
    if "per_geometry" not in doc or len(doc["per_geometry"]) != 135:
        raise ValueError("unexpected thermal geometry specification structure")
    return doc


def geometry_entry(spec: dict, geom_id: int) -> dict:
    entries = spec["per_geometry"]
    if not 0 <= geom_id < len(entries):
        raise IndexError(f"geometry id out of range: {geom_id}")
    entry = entries[geom_id]
    if int(entry["geom"]) != geom_id:
        raise ValueError(f"geometry table mismatch at id {geom_id}")
    return entry


def polygon_vertices(entry: dict) -> np.ndarray:
    """Return the regular-polygon vertices (counter-clockwise).

    The specification stores the *edge-normal* phase (``_signed`` convention in
    scripts/gen_thermal_geometries.py); vertices are therefore shifted by pi/n.
    """
    n = int(entry["n"])
    radius = float(entry["R"])
    phase = math.radians(float(entry["phase_deg"]))
    centre = np.asarray(entry["centre"], dtype=float)
    angles = phase + math.pi / n + 2.0 * math.pi * np.arange(n) / n
    return centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])


def polygon_apothem(entry: dict) -> float:
    return float(entry["R"]) * math.cos(math.pi / int(entry["n"]))


def inner_radius(theta: float | np.ndarray, entry: dict) -> float | np.ndarray:
    """Ray distance from the centre to the inner polygon boundary."""
    n = int(entry["n"])
    radius = float(entry["R"])
    phase = math.radians(float(entry["phase_deg"]))
    theta_arr = np.asarray(theta, dtype=float)
    k = np.arange(n)
    normals = phase + 2.0 * math.pi * k / n
    proj = np.max(np.cos(theta_arr[..., None] - normals[None, :]), axis=-1)
    return polygon_apothem(entry) / proj


def outer_radius(theta: float | np.ndarray) -> float | np.ndarray:
    """Ray distance from the centre to the boundary of [-1,1]^2."""
    theta_arr = np.asarray(theta, dtype=float)
    return 1.0 / np.maximum(np.abs(np.cos(theta_arr)), np.abs(np.sin(theta_arr)))


def _critical_angles(entry: dict) -> np.ndarray:
    n = int(entry["n"])
    phase = math.radians(float(entry["phase_deg"]))
    vertex = phase + math.pi / n + 2.0 * math.pi * np.arange(n) / n
    corners = np.array([0.25, 0.75, 1.25, 1.75]) * math.pi
    return np.mod(np.concatenate([vertex, corners]), 2.0 * math.pi)


def _unique_angles(n_theta: int, entry: dict) -> np.ndarray:
    base = np.linspace(0.0, 2.0 * math.pi, int(n_theta), endpoint=False)
    angles = np.mod(np.concatenate([base, _critical_angles(entry)]),
                    2.0 * math.pi)
    angles = np.round(angles, 12)
    angles = np.unique(angles)
    return angles


def _triangle_area(tri: np.ndarray) -> np.ndarray:
    a = tri[:, 1] - tri[:, 0]
    b = tri[:, 2] - tri[:, 0]
    return 0.5 * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])


def _triangle_angles(tri: np.ndarray) -> np.ndarray:
    """Minimum angle (degrees) for every triangle."""
    edges = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1],
                      tri[:, 0] - tri[:, 2]], axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    out = np.empty(len(tri))
    for index, (a, b, c) in enumerate(edges):
        la, lb, lc = lengths[index]
        # angle opposite side a
        cos_a = np.clip((lb ** 2 + lc ** 2 - la ** 2) / (2 * lb * lc), -1.0, 1.0)
        cos_b = np.clip((la ** 2 + lc ** 2 - lb ** 2) / (2 * la * lc), -1.0, 1.0)
        out[index] = math.degrees(math.acos(max(cos_a, -1.0)))
        out[index] = min(out[index], math.degrees(math.acos(max(cos_b, -1.0))))
        # third angle
        cos_c = np.clip((la ** 2 + lb ** 2 - lc ** 2) / (2 * la * lb), -1.0, 1.0)
        out[index] = min(out[index], math.degrees(math.acos(max(cos_c, -1.0))))
    return out


def build_polar_mesh(entry: dict, *, target_h: float | None = None,
                     n_theta: int | None = None,
                     n_radial: int | None = None,
                     radial_distribution: str = "cosine") -> dict:
    """Build a conforming triangle mesh of square-minus-polygon.

    Radii are taken along rays from the centre; inner ring = polygon boundary,
    outer ring = square boundary.  Polygon vertices and square corners are
    inserted as ray angles, so both boundaries are represented exactly.
    """
    inner_perimeter = int(entry["n"]) * float(entry["side"])
    if n_theta is None:
        if target_h is None:
            raise ValueError("either target_h or n_theta is required")
        n_theta = max(12, int(math.ceil((8.0 + inner_perimeter) / target_h)))
    if n_radial is None:
        if target_h is None:
            raise ValueError("either target_h or n_radial is required")
        max_thickness = 1.0 - polygon_apothem(entry) * math.cos(math.pi / int(entry["n"]))
        n_radial = max(3, int(math.ceil((1.0 + max_thickness) / target_h)))
    angles = _unique_angles(int(n_theta), entry)
    n_theta = len(angles)
    n_radial = int(n_radial)
    if radial_distribution == "cosine":
        t = 0.5 * (1.0 - np.cos(math.pi * np.arange(n_radial + 1) / n_radial))
    elif radial_distribution == "uniform":
        t = np.linspace(0.0, 1.0, n_radial + 1)
    else:
        raise ValueError(f"unknown radial_distribution: {radial_distribution}")
    radii_in = np.asarray(inner_radius(angles, entry), dtype=float)
    radii_out = np.asarray(outer_radius(angles), dtype=float)
    if np.any(radii_out <= radii_in):
        raise ValueError("outer boundary is not outside the inner polygon")
    nodes = np.empty((n_theta * (n_radial + 1), 2), dtype=float)
    for i, theta in enumerate(angles):
        radius = radii_in[i] + t * (radii_out[i] - radii_in[i])
        nodes[i * (n_radial + 1):(i + 1) * (n_radial + 1)] = np.column_stack([
            radius * math.cos(theta), radius * math.sin(theta)])
    triangles = []
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        for j in range(n_radial):
            a = i * (n_radial + 1) + j
            b = i2 * (n_radial + 1) + j
            c = i2 * (n_radial + 1) + j + 1
            d = i * (n_radial + 1) + j + 1
            if (i + j) % 2 == 0:
                triangles.extend([(a, b, c), (a, c, d)])
            else:
                triangles.extend([(a, b, d), (b, c, d)])
    triangles = np.asarray(triangles, dtype=np.int64)
    tri_xyz = nodes[triangles]
    areas = _triangle_area(tri_xyz)
    if np.any(areas <= 0):
        flip = areas < 0
        triangles[flip] = triangles[flip][:, [0, 2, 1]]
        tri_xyz = nodes[triangles]
        areas = _triangle_area(tri_xyz)
    if np.any(areas <= 0):
        raise ValueError("mesh contains non-positive-area triangles")
    boundary_side = np.full(len(nodes), -1, dtype=np.int8)
    boundary_side[0 * (n_radial + 1):: (n_radial + 1)] = 0      # inner
    # outer ring: last node in each ray
    outer_ids = np.array([i * (n_radial + 1) + n_radial
                          for i in range(n_theta)], dtype=np.int64)
    boundary_side[outer_ids] = 1
    boundary_edges = set()
    for tri in triangles:
        for first, second in ((tri[0], tri[1]), (tri[1], tri[2]),
                              (tri[2], tri[0])):
            if boundary_side[first] >= 0 and boundary_side[second] >= 0:
                boundary_edges.add(tuple(sorted((int(first), int(second)))))
    boundary_edges = sorted(boundary_edges)
    polygon_area = 0.5 * int(entry["n"]) * float(entry["R"]) ** 2 * math.sin(
        2.0 * math.pi / int(entry["n"]))
    expected_area = 4.0 - polygon_area
    mesh = {
        "nodes": nodes,
        "triangles": triangles,
        "boundary_side": boundary_side,
        "boundary_edges": boundary_edges,
        "n_theta": n_theta,
        "n_radial": n_radial,
        "angles": angles,
        "radial_distribution": radial_distribution,
        "h_max": float(np.max(np.linalg.norm(
            np.stack([nodes[triangles[:, 1]] - nodes[triangles[:, 0]],
                      nodes[triangles[:, 2]] - nodes[triangles[:, 1]],
                      nodes[triangles[:, 0]] - nodes[triangles[:, 2]]],
                     axis=1), axis=2))),
        "area": float(areas.sum()),
        "expected_area": float(expected_area),
        "area_relative_error": float(abs(areas.sum() - expected_area) / expected_area),
        "min_triangle_angle_deg": float(np.min(_triangle_angles(tri_xyz))),
        "max_edge_ratio": float(np.max([
            np.max(np.linalg.norm(
                np.stack([nodes[t[1]] - nodes[t[0]], nodes[t[2]] - nodes[t[1]],
                          nodes[t[0]] - nodes[t[2]]]), axis=1))
            / max(np.min(np.linalg.norm(
                np.stack([nodes[t[1]] - nodes[t[0]], nodes[t[2]] - nodes[t[1]],
                          nodes[t[0]] - nodes[t[2]]]), axis=1)), 1e-300)
            for t in triangles])),
        "n_nodes": int(len(nodes)),
        "n_triangles": int(len(triangles)),
    }
    return mesh


def _point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to a set of segments (M x S)."""
    starts = segments[:, 0][None, :, :]
    ends = segments[:, 1][None, :, :]
    direction = ends - starts
    denom = np.sum(direction * direction, axis=2)
    denom = np.where(denom > 0, denom, 1.0)
    rel = points[:, None, :] - starts
    t = np.clip(np.sum(rel * direction, axis=2) / denom, 0.0, 1.0)
    closest = starts + t[..., None] * direction
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def polygon_segments(entry: dict) -> np.ndarray:
    vertices = polygon_vertices(entry)
    return np.stack([vertices, np.roll(vertices, -1, axis=0)], axis=1)


def square_segments() -> np.ndarray:
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    return np.stack([corners, np.roll(corners, -1, axis=0)], axis=1)


def load_boundary_points(data_path: str, geom_id: int,
                         boundary_n: int = BOUNDARY_N) -> dict:
    arr = np.load(data_path)
    if arr.shape != (135, 5000, 6):
        raise ValueError(f"unexpected HeatTransferData shape: {arr.shape}")
    points = arr[geom_id, :boundary_n, :2]
    outer_mask = (np.isclose(np.abs(points[:, 0]), 1.0, atol=1e-9)
                  | np.isclose(np.abs(points[:, 1]), 1.0, atol=1e-9))
    return {"points": points, "outer_mask": outer_mask,
            "outer": points[outer_mask], "inner": points[~outer_mask],
            "values": arr[geom_id, :boundary_n],
            "all": arr[geom_id]}


def _signed_polygon_distance(points: np.ndarray, n: int, radius: float,
                             phase: float, centre: np.ndarray) -> np.ndarray:
    """Signed distance to a regular n-gon with edge-normal phase ``phase``."""
    k = np.arange(n)
    normals = np.column_stack([np.cos(phase + 2.0 * math.pi * k / n),
                               np.sin(phase + 2.0 * math.pi * k / n)])
    proj = (points - centre) @ normals.T
    nearest = np.argmax(proj, axis=1)
    return proj[np.arange(len(points)), nearest] - radius * math.cos(math.pi / n)


def fit_polygon_from_points(points: np.ndarray, n: int) -> dict:
    """Independently fit a regular n-gon to the original inner boundary points."""
    from scipy.optimize import minimize

    centre0 = points.mean(axis=0)
    best = None
    for phase_deg in np.arange(0.0, 360.0 / n, 0.5):
        phase = math.radians(float(phase_deg))
        k = np.arange(n)
        normals = np.column_stack([np.cos(phase + 2.0 * math.pi * k / n),
                                   np.sin(phase + 2.0 * math.pi * k / n)])
        proj = (points - centre0) @ normals.T
        nearest = np.argmax(proj, axis=1)
        radius0 = float(np.mean(proj[np.arange(len(points)), nearest])
                        / math.cos(math.pi / n))
        if radius0 <= 0:
            continue
        residual = _signed_polygon_distance(points, n, radius0, phase, centre0)
        value = float(np.mean(residual ** 2))
        if best is None or value < best[0]:
            best = (value, radius0, phase)
    if best is None:
        raise RuntimeError("polygon fit failed to find a positive radius")
    def objective(x):
        residual = _signed_polygon_distance(
            points, n, x[0], x[1], np.asarray(x[2:], dtype=float))
        return float(np.mean(residual ** 2))
    result = minimize(objective, [best[1], best[2], centre0[0], centre0[1]],
                      method="Nelder-Mead",
                      options={"xatol": 1e-14, "fatol": 1e-24, "maxiter": 8000})
    radius, phase = float(result.x[0]), float(result.x[1])
    centre = np.asarray(result.x[2:], dtype=float)
    return {"n": int(n), "rms": float(math.sqrt(max(result.fun, 0.0))),
            "radius": radius,
            "side": float(2.0 * radius * math.sin(math.pi / n)),
            "phase_deg": float(math.degrees(phase) % (360.0 / n)),
            "centre": centre.tolist()}


def verify_geometry(entry: dict, boundary: dict) -> dict:
    outer = boundary["outer"]
    inner = boundary["inner"]
    inner_dev = _point_segment_distances(inner, polygon_segments(entry))
    outer_dev = _point_segment_distances(outer, square_segments())
    vertices = polygon_vertices(entry)
    centre = np.asarray(entry["centre"], dtype=float)
    fitted = fit_polygon_from_points(inner, int(entry["n"]))
    return {
        "geom": int(entry["geom"]),
        "n": int(entry["n"]),
        "outer_count": int(len(outer)),
        "inner_count": int(len(inner)),
        "outer_hausdorff_like_max": float(np.max(outer_dev)),
        "inner_point_to_polygon_rms": float(np.sqrt(np.mean(inner_dev ** 2))),
        "inner_point_to_polygon_max": float(np.max(inner_dev)),
        "polygon_centre_norm": float(np.linalg.norm(centre)),
        "polygon_radius": float(entry["R"]),
        "polygon_side": float(entry["side"]),
        "polygon_phase_deg": float(entry["phase_deg"]),
        "independent_polygon_fit": fitted,
        "fit_radius_abs_error": float(abs(fitted["radius"] - entry["R"])),
        "fit_side_abs_error": float(abs(fitted["side"] - entry["side"])),
        "fit_phase_abs_error_deg": float(abs(
            (fitted["phase_deg"] - entry["phase_deg"] + 180.0 / int(entry["n"]))
            % (360.0 / int(entry["n"])) - 180.0 / int(entry["n"]))),
        "vertices": vertices.tolist(),
    }


def boundary_value_checks(boundary: dict) -> dict:
    values = boundary["values"]
    velocity = values[:, 2:4]
    temperature = values[:, 5]
    outer = boundary["outer_mask"]
    inner = ~outer
    return {
        "wall_velocity_max": float(np.max(np.abs(velocity))),
        "inner_temperature_min": float(np.min(temperature[inner])),
        "inner_temperature_max": float(np.max(temperature[inner])),
        "outer_temperature_min": float(np.min(temperature[outer])),
        "outer_temperature_max": float(np.max(temperature[outer])),
        "inner_count": int(inner.sum()),
        "outer_count": int(outer.sum()),
    }


def split_geometry_ids(split_path: str = SPLIT_CONFIG_PATH,
                       protocol: str = "protocol_A") -> dict:
    split = load_json(split_path)
    ids = split[protocol]
    return {name: [int(x) for x in values] for name, values in ids.items()}
