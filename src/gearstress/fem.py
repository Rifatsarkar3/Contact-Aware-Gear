"""CalculiX/Gmsh backends for local gear-contact feasibility studies.

The original model is deliberately a local contact surrogate: an involute tooth
sector pressed by a cylindrical mating-flank approximation.  The mating-tooth
backend added below uses two deformable involute sectors and is the scientific
smoke gate before generating a full transient dataset.  Both use plane strain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess

import gmsh
import numpy as np
from scipy.interpolate import griddata


@dataclass(frozen=True)
class ContactCase:
    module_mm: float = 2.0
    teeth: int = 24
    pressure_angle_deg: float = 20.0
    face_width_mm: float = 1.0
    root_clearance_module: float = 0.25
    indenter_radius_mm: float = 1.0
    contact_fraction: float = 0.52
    initial_gap_mm: float = 0.005
    indentation_mm: float = 0.025
    friction: float = 0.08
    youngs_modulus_mpa: float = 210_000.0
    poisson_ratio: float = 0.30
    mesh_size_mm: float = 0.65

    @property
    def pitch_radius_mm(self) -> float:
        return 0.5 * self.module_mm * self.teeth

    @property
    def base_radius_mm(self) -> float:
        return self.pitch_radius_mm * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def outer_radius_mm(self) -> float:
        return self.pitch_radius_mm + self.module_mm

    @property
    def root_radius_mm(self) -> float:
        return self.pitch_radius_mm - self.module_mm * (1.0 + self.root_clearance_module)


@dataclass(frozen=True)
class GearPairCase:
    """Quasi-static contact case for two identical deformable involute sectors."""

    module_mm: float = 2.0
    teeth: int = 24
    pressure_angle_deg: float = 20.0
    face_width_mm: float = 1.0
    root_clearance_module: float = 0.25
    contact_fraction: float = 0.52
    initial_gap_mm: float = 0.005
    indentation_mm: float = 0.025
    friction: float = 0.08
    youngs_modulus_mpa: float = 210_000.0
    poisson_ratio: float = 0.30
    mesh_size_mm: float = 0.12
    contact_penalty_factor: float = 50.0

    @property
    def pitch_radius_mm(self) -> float:
        return 0.5 * self.module_mm * self.teeth

    @property
    def base_radius_mm(self) -> float:
        return self.pitch_radius_mm * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def outer_radius_mm(self) -> float:
        return self.pitch_radius_mm + self.module_mm

    @property
    def root_radius_mm(self) -> float:
        return self.pitch_radius_mm - self.module_mm * (1.0 + self.root_clearance_module)


@dataclass(frozen=True)
class GearPairTransientCase:
    """Kinematically-driven transient dynamic contact case.

    Two deformable involute sectors, meshed once at ``phase_start``, then
    swept continuously to ``phase_end`` over ``n_time_steps`` chained
    ``*DYNAMIC`` sub-steps with real material density.  Driver root stays
    fixed (local co-rotating-frame simplification, same as ``GearPairCase``);
    only the mating root is re-targeted at each time sample.
    """

    module_mm: float = 2.0
    teeth: int = 24
    pressure_angle_deg: float = 20.0
    face_width_mm: float = 1.0
    root_clearance_module: float = 0.25
    phase_start: float = 0.38
    phase_end: float = 0.62
    initial_gap_mm: float = 0.005
    indentation_mm: float = 0.025
    friction: float = 0.08
    youngs_modulus_mpa: float = 210_000.0
    poisson_ratio: float = 0.30
    density_tonne_per_mm3: float = 7.85e-9
    mesh_size_mm: float = 0.12
    n_time_steps: int = 6
    mean_speed_rpm: float = 3000.0
    speed_ripple_fraction: float = 0.0

    @property
    def pitch_radius_mm(self) -> float:
        return 0.5 * self.module_mm * self.teeth

    @property
    def base_radius_mm(self) -> float:
        return self.pitch_radius_mm * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def outer_radius_mm(self) -> float:
        return self.pitch_radius_mm + self.module_mm

    @property
    def root_radius_mm(self) -> float:
        return self.pitch_radius_mm - self.module_mm * (1.0 + self.root_clearance_module)

    def as_gear_pair_case(self, contact_fraction: float) -> "GearPairCase":
        """Project this case onto the existing static-case geometry fields at one phase."""
        return GearPairCase(
            module_mm=self.module_mm,
            teeth=self.teeth,
            pressure_angle_deg=self.pressure_angle_deg,
            face_width_mm=self.face_width_mm,
            root_clearance_module=self.root_clearance_module,
            contact_fraction=contact_fraction,
            initial_gap_mm=self.initial_gap_mm,
            indentation_mm=self.indentation_mm,
            friction=self.friction,
            youngs_modulus_mpa=self.youngs_modulus_mpa,
            poisson_ratio=self.poisson_ratio,
            mesh_size_mm=self.mesh_size_mm,
        )


def _rotate_to_vertical(points: np.ndarray) -> np.ndarray:
    return np.column_stack((-points[:, 1], points[:, 0]))


def involute_flank(case: ContactCase | GearPairCase, samples: int = 20) -> np.ndarray:
    """Return the right involute flank from base circle to addendum circle."""
    rb = case.base_radius_mm
    alpha = math.radians(case.pressure_angle_deg)
    half_tooth = math.pi / (2.0 * case.teeth)
    base_angle = half_tooth - (math.tan(alpha) - alpha)
    # Uniform-radius sampling avoids nearly coincident points at the involute
    # origin, where the derivative with respect to the involute parameter is zero.
    radius = np.linspace(rb, case.outer_radius_mm, samples)
    t = np.sqrt((radius / rb) ** 2 - 1.0)
    x = rb * (np.cos(t) + t * np.sin(t))
    y = rb * (np.sin(t) - t * np.cos(t))
    c, s = math.cos(base_angle), math.sin(base_angle)
    flank = np.column_stack((c * x - s * y, s * x + c * y))
    return _rotate_to_vertical(flank)


def conjugate_involute_flank(case: GearPairCase, samples: int = 80) -> np.ndarray:
    """Return a standard external-gear flank that thins toward the addendum."""
    rb = case.base_radius_mm
    alpha = math.radians(case.pressure_angle_deg)
    half_tooth = math.pi / (2.0 * case.teeth)
    base_angle = half_tooth + (math.tan(alpha) - alpha)
    radius = np.linspace(rb, case.outer_radius_mm, samples)
    t = np.sqrt((radius / rb) ** 2 - 1.0)
    x = rb * (np.cos(t) + t * np.sin(t))
    y = -rb * (np.sin(t) - t * np.cos(t))
    c, s = math.cos(base_angle), math.sin(base_angle)
    flank = np.column_stack((c * x - s * y, s * x + c * y))
    return _rotate_to_vertical(flank)


def tooth_polygon(case: ContactCase | GearPairCase, flank_samples: int = 20, arc_samples: int = 12) -> np.ndarray:
    """Create a closed, polygonal involute-tooth sector in counter-clockwise order."""
    left = involute_flank(case, flank_samples)
    right = left.copy()
    right[:, 0] *= -1.0
    rb = case.base_radius_mm
    rr = case.root_radius_mm
    sector = math.pi / case.teeth
    base_angle_vertical = math.atan2(right[0, 1], right[0, 0])
    # Angles are measured from +x in the vertically oriented coordinate system.
    left_base_angle = math.atan2(left[0, 1], left[0, 0])
    right_base_angle = base_angle_vertical
    left_sector_angle = math.pi / 2.0 + sector
    right_sector_angle = math.pi / 2.0 - sector
    inner_radius = rr - 0.75 * case.module_mm
    left_inner = np.array([inner_radius * math.cos(left_sector_angle), inner_radius * math.sin(left_sector_angle)])
    right_inner = np.array([inner_radius * math.cos(right_sector_angle), inner_radius * math.sin(right_sector_angle)])
    left_root_angles = np.linspace(left_sector_angle, left_base_angle, 6)
    right_root_angles = np.linspace(right_base_angle, right_sector_angle, 6)
    left_root_arc = rr * np.column_stack((np.cos(left_root_angles), np.sin(left_root_angles)))
    right_root_arc = rr * np.column_stack((np.cos(right_root_angles), np.sin(right_root_angles)))

    theta_left_tip = math.atan2(left[-1, 1], left[-1, 0])
    theta_right_tip = math.atan2(right[-1, 1], right[-1, 0])
    tip_angles = np.linspace(theta_left_tip, theta_right_tip, arc_samples)
    tip = case.outer_radius_mm * np.column_stack((np.cos(tip_angles), np.sin(tip_angles)))
    inner_angles = np.linspace(right_sector_angle, left_sector_angle, arc_samples)
    inner_arc = inner_radius * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    polygon = np.vstack(
        (
            left_inner,
            left_root_arc,
            left,
            tip[1:-1],
            right[::-1],
            right_root_arc,
            right_inner,
            inner_arc[1:-1],
        )
    )
    return polygon


def gear_tooth_polygon(case: GearPairCase, flank_samples: int = 80, arc_samples: int = 24) -> np.ndarray:
    """Create a corrected external involute-tooth sector for mating contact."""
    left = conjugate_involute_flank(case, flank_samples)
    right = left.copy()
    right[:, 0] *= -1.0
    rr = case.root_radius_mm
    sector = math.pi / case.teeth
    left_base_angle = math.atan2(left[0, 1], left[0, 0])
    right_base_angle = math.atan2(right[0, 1], right[0, 0])
    left_sector_angle = math.pi / 2.0 + sector
    right_sector_angle = math.pi / 2.0 - sector
    inner_radius = rr - 0.75 * case.module_mm
    left_inner = np.array([inner_radius * math.cos(left_sector_angle), inner_radius * math.sin(left_sector_angle)])
    right_inner = np.array([inner_radius * math.cos(right_sector_angle), inner_radius * math.sin(right_sector_angle)])
    left_root_angles = np.linspace(left_sector_angle, left_base_angle, 12)
    right_root_angles = np.linspace(right_base_angle, right_sector_angle, 12)
    left_root_arc = rr * np.column_stack((np.cos(left_root_angles), np.sin(left_root_angles)))
    right_root_arc = rr * np.column_stack((np.cos(right_root_angles), np.sin(right_root_angles)))
    theta_left_tip = math.atan2(left[-1, 1], left[-1, 0])
    theta_right_tip = math.atan2(right[-1, 1], right[-1, 0])
    tip_angles = np.linspace(theta_left_tip, theta_right_tip, arc_samples)
    tip = case.outer_radius_mm * np.column_stack((np.cos(tip_angles), np.sin(tip_angles)))
    inner_angles = np.linspace(right_sector_angle, left_sector_angle, arc_samples)
    inner_arc = inner_radius * np.column_stack((np.cos(inner_angles), np.sin(inner_angles)))
    return np.vstack(
        (
            left_inner,
            left_root_arc,
            left,
            tip[1:-1],
            right[::-1],
            right_root_arc,
            right_inner,
            inner_arc[1:-1],
        )
    )


def flank_point_and_normal(
    case: ContactCase | GearPairCase,
    fraction: float,
    side: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a polygon flank vertex and its outward unit normal."""
    flank = involute_flank(case, 20)
    if side == "right":
        flank = flank.copy()
        flank[:, 0] *= -1.0
    elif side != "left":
        raise ValueError("side must be 'left' or 'right'")
    index = int(round(np.clip(fraction, 0.05, 0.95) * (len(flank) - 1)))
    index = int(np.clip(index, 1, len(flank) - 2))
    point = flank[index]
    tangent = flank[index + 1] - flank[index - 1]
    tangent /= np.linalg.norm(tangent)
    normal = np.array((tangent[1], -tangent[0]))
    if normal[0] * point[0] < 0:
        normal *= -1.0
    return point, normal


def contact_point_and_normal(case: ContactCase | GearPairCase) -> tuple[np.ndarray, np.ndarray]:
    # Use the same flank discretization as the geometry so the initial clearance
    # is measured from an actual boundary vertex, not the continuous involute.
    return flank_point_and_normal(case, case.contact_fraction, side="left")


def mating_tooth_transform(case: GearPairCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return equal-gear meshing kinematics in the driver tooth frame.

    The mating gear is half a tooth pitch out of phase and counter-rotates at a
    1:1 ratio.  ``contact_fraction`` parameterizes a small mesh-angle window for
    this scientific smoke case.  A sub-mesh translation then enforces the stated
    initial clearance without changing the relative rotation.
    """
    phase = float(np.clip((case.contact_fraction - 0.38) / 0.24, 0.0, 1.0))
    driver_angle = math.radians(-3.0 + 6.0 * phase)
    tooth_pitch = 2.0 * math.pi / case.teeth
    relative_angle = math.pi + 0.5 * tooth_pitch - 2.0 * driver_angle
    rotation = np.asarray(
        ((math.cos(relative_angle), -math.sin(relative_angle)),
         (math.sin(relative_angle), math.cos(relative_angle))),
        dtype=np.float64,
    )
    centre_global = np.asarray((0.0, 2.0 * case.pitch_radius_mm))
    co_rotate = np.asarray(
        ((math.cos(-driver_angle), -math.sin(-driver_angle)),
         (math.sin(-driver_angle), math.cos(-driver_angle))),
        dtype=np.float64,
    )
    translation = co_rotate @ centre_global
    polygon = gear_tooth_polygon(case)
    mating = polygon @ rotation.T + translation
    distances = np.linalg.norm(polygon[:, None, :] - mating[None, :, :], axis=2)
    driver_index, mating_index = np.unravel_index(np.argmin(distances), distances.shape)
    point = polygon[driver_index]
    separation = mating[mating_index] - point
    distance = float(np.linalg.norm(separation))
    if distance <= 1e-10:
        raise RuntimeError("mating-tooth geometry has zero initial clearance")
    normal = separation / distance
    translation = translation + normal * (case.initial_gap_mm - distance)
    return rotation, translation, point, normal


def mating_node_targets(
    case: "GearPairTransientCase",
    reference_node_xy: dict[int, np.ndarray],
    phase_start_rotation: np.ndarray,
    phase_start_translation: np.ndarray,
    phase: float,
    load_fraction: float,
) -> dict[int, np.ndarray]:
    """Absolute (x, y) target for each mating-root node at a later phase/load level.

    ``reference_node_xy`` holds each node's meshed position at ``case.phase_start``.
    """
    rotation_k, translation_k, _, normal_k = mating_tooth_transform(case.as_gear_pair_case(phase))
    inverse_rotation_0 = phase_start_rotation.T
    targets: dict[int, np.ndarray] = {}
    for node_id, xy0 in reference_node_xy.items():
        x_ref = inverse_rotation_0 @ (xy0 - phase_start_translation)
        targets[node_id] = rotation_k @ x_ref + translation_k - normal_k * (
            case.indentation_mm * load_fraction
        )
    return targets


def transient_time_samples(
    phase_start: float,
    phase_end: float,
    n_steps: int,
    mean_speed_rpm: float,
    speed_ripple_fraction: float = 0.0,
) -> np.ndarray:
    """Return ``(n_steps + 1, 2)`` array of ``(time_s, phase)`` at equal angular
    increments of the driver, under a possibly time-varying angular speed.

    ``phase`` maps linearly to driver angle exactly as in
    ``mating_tooth_transform`` (``angle_deg = -3 + 6 * phase``).  Equal
    angle increments are inverted to real (generally non-uniform when
    ``speed_ripple_fraction != 0``) time instants by bisecting the closed-form
    integral of ``omega(t) = omega_mean * (1 + speed_ripple_fraction *
    sin(2*pi*t/period))``, where ``period`` is the nominal constant-speed
    sweep duration used only to set the ripple's own oscillation rate.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if not (0.0 <= speed_ripple_fraction < 1.0):
        raise ValueError("speed_ripple_fraction must be in [0, 1)")
    if mean_speed_rpm <= 0.0:
        raise ValueError("mean_speed_rpm must be positive")
    total_angle_rad = math.radians(6.0 * (phase_end - phase_start))
    if total_angle_rad <= 0.0:
        raise ValueError("phase_end must exceed phase_start")
    omega_mean = mean_speed_rpm * 2.0 * math.pi / 60.0
    nominal_period = total_angle_rad / omega_mean

    def angle_of_time(t: float) -> float:
        if speed_ripple_fraction == 0.0:
            return omega_mean * t
        return omega_mean * t + omega_mean * speed_ripple_fraction * nominal_period / (
            2.0 * math.pi
        ) * (1.0 - math.cos(2.0 * math.pi * t / nominal_period))

    upper_bound = nominal_period * 4.0 / max(1e-9, (1.0 - speed_ripple_fraction))
    samples = np.zeros((n_steps + 1, 2), dtype=np.float64)
    samples[0] = (0.0, phase_start)
    for k in range(1, n_steps + 1):
        target_angle = total_angle_rad * k / n_steps
        lo, hi = 0.0, upper_bound
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if angle_of_time(mid) < target_angle:
                lo = mid
            else:
                hi = mid
        t_k = 0.5 * (lo + hi)
        samples[k] = (t_k, phase_start + (phase_end - phase_start) * k / n_steps)
    return samples


def _surface_edges(
    area_elements: dict[int, np.ndarray], curve_tags: list[int]
) -> list[tuple[int, str]]:
    """Map Gmsh boundary lines to CalculiX triangular plane-element faces."""
    edge_map: dict[tuple[int, int], tuple[int, str]] = {}
    edge_indices = {
        "S1": (0, 1),
        "S2": (1, 2),
        "S3": (2, 0),
    }
    for element_id, nodes in area_elements.items():
        for face, idx in edge_indices.items():
            edge_map[tuple(sorted(int(nodes[i]) for i in idx))] = (element_id, face)
    result: set[tuple[int, str]] = set()
    for curve_tag in curve_tags:
        types, _, node_blocks = gmsh.model.mesh.getElements(1, curve_tag)
        for element_type, flat_nodes in zip(types, node_blocks):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if nodes_per_element != 2:
                continue
            for nodes in np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 2):
                match = edge_map.get(tuple(sorted(int(v) for v in nodes)))
                if match is not None:
                    result.add(match)
    return sorted(result)


def build_case(case: ContactCase, output_dir: Path) -> Path:
    """Mesh a contact case and write a self-contained CalculiX input deck."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("gear_contact_smoke")
        polygon = tooth_polygon(case)
        point_tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in polygon]
        line_tags = [
            gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
            for i in range(len(point_tags))
        ]
        loop = gmsh.model.occ.addCurveLoop(line_tags)
        tooth_surface = gmsh.model.occ.addPlaneSurface([loop])
        contact_point, normal = contact_point_and_normal(case)
        center = contact_point + normal * (case.indenter_radius_mm + case.initial_gap_mm)
        indenter_surface = gmsh.model.occ.addDisk(
            float(center[0]),
            float(center[1]),
            0.0,
            case.indenter_radius_mm,
            case.indenter_radius_mm,
        )
        gmsh.model.occ.synchronize()
        indenter_boundary = [
            tag for dim, tag in gmsh.model.getBoundary([(2, indenter_surface)], oriented=False) if dim == 1
        ]
        indenter_points = [
            (dim, tag)
            for dim, tag in gmsh.model.getBoundary([(2, indenter_surface)], oriented=False, recursive=True)
            if dim == 0
        ]

        gmsh.option.setNumber("Mesh.MeshSizeMin", case.mesh_size_mm * 0.55)
        gmsh.option.setNumber("Mesh.MeshSizeMax", case.mesh_size_mm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.setSize(indenter_points, case.mesh_size_mm * 0.22)
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        nodes = {int(tag): coord for tag, coord in zip(node_tags, coordinates)}

        element_sets: dict[str, dict[int, np.ndarray]] = {}
        for name, area in (("TOOTH", tooth_surface), ("INDENTER", indenter_surface)):
            types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(2, area)
            elements: dict[int, np.ndarray] = {}
            for element_type, tags, flat_nodes in zip(types, tag_blocks, node_blocks):
                _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if nodes_per_element != 3:
                    continue
                shaped = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 3)
                for row in shaped:
                    a, b, c = (nodes[int(tag)][:2] for tag in row)
                    signed_twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if signed_twice_area < 0:
                        row[1], row[2] = row[2], row[1]
                elements.update({int(tag): row for tag, row in zip(tags, shaped)})
            if not elements:
                element_debug = [
                    (int(element_type), gmsh.model.mesh.getElementProperties(element_type)[0], len(tags))
                    for element_type, tags in zip(types, tag_blocks)
                ]
                raise RuntimeError(f"no linear triangles generated for {name}: {element_debug}")
            element_sets[name] = elements
        all_elements = {**element_sets["TOOTH"], **element_sets["INDENTER"]}
        tooth_faces = _surface_edges(all_elements, line_tags)
        indenter_faces = _surface_edges(all_elements, indenter_boundary)
        if not tooth_faces or not indenter_faces:
            raise RuntimeError("failed to map contact surfaces to tetrahedron faces")

        rr = case.root_radius_mm
        tooth_node_ids = sorted({int(v) for row in element_sets["TOOTH"].values() for v in row})
        indenter_node_ids = sorted({int(v) for row in element_sets["INDENTER"].values() for v in row})
        root_nodes = [
            tag for tag in tooth_node_ids if np.linalg.norm(nodes[tag][:2]) <= rr + case.mesh_size_mm * 0.45
        ]
        if len(root_nodes) < 3:
            raise RuntimeError("root support set is unexpectedly small")

        def ids_block(values: list[int]) -> str:
            return "\n".join(", ".join(map(str, values[i : i + 16])) for i in range(0, len(values), 16))

        lines = ["*HEADING", "Single-tooth contact FEM software smoke"]
        lines.append("*NODE")
        lines.extend(f"{tag}, {xyz[0]:.10g}, {xyz[1]:.10g}, {xyz[2]:.10g}" for tag, xyz in sorted(nodes.items()))
        for name, elements in element_sets.items():
            lines.append(f"*ELEMENT, TYPE=CPE3, ELSET={name}")
            lines.extend(f"{tag}, " + ", ".join(map(str, row)) for tag, row in sorted(elements.items()))
        lines.extend(("*NSET, NSET=ROOT", ids_block(root_nodes)))
        lines.extend(("*NSET, NSET=TOOTHNODES", ids_block(tooth_node_ids)))
        lines.extend(("*NSET, NSET=INDENTERNODES", ids_block(indenter_node_ids)))
        lines.append("*SURFACE, NAME=TOOTHSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in tooth_faces)
        lines.append("*SURFACE, NAME=INDENTERSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in indenter_faces)
        lines.extend(
            (
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.youngs_modulus_mpa:.10g}, {case.poisson_ratio:.10g}",
                "*SOLID SECTION, ELSET=TOOTH, MATERIAL=STEEL",
                "*SOLID SECTION, ELSET=INDENTER, MATERIAL=STEEL",
                "*SURFACE INTERACTION, NAME=CONTACTINT",
                "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                f"{50.0 * case.youngs_modulus_mpa:.10g}",
                "*FRICTION",
                f"{case.friction:.10g}, {0.5 * case.youngs_modulus_mpa:.10g}",
                "*CONTACT PAIR, INTERACTION=CONTACTINT, TYPE=SURFACE TO SURFACE, ADJUST=0.03",
                "INDENTERSURF, TOOTHSURF",
                "*BOUNDARY",
                "ROOT, 1, 3, 0.0",
                "*STEP, NLGEOM",
                "*STATIC",
                "0.05, 1.0, 1e-06, 0.1",
                "*BOUNDARY",
                f"INDENTERNODES, 1, 1, {-normal[0] * case.indentation_mm:.10g}",
                f"INDENTERNODES, 2, 2, {-normal[1] * case.indentation_mm:.10g}",
                "*NODE FILE",
                "U, RF",
                "*EL FILE",
                "S, E",
                "*CONTACT FILE",
                "CDIS, CSTR, CELS",
                "*END STEP",
            )
        )
        deck = output_dir / "case.inp"
        deck.write_text("\n".join(lines) + "\n", encoding="ascii")
        metadata = {**asdict(case), "contact_point_mm": contact_point.tolist(), "contact_normal": normal.tolist(), "nodes": len(nodes), "elements": len(all_elements), "root_nodes": len(root_nodes), "tooth_contact_faces": len(tooth_faces), "indenter_contact_faces": len(indenter_faces)}
        (output_dir / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        tooth_element_ids = np.asarray(sorted(element_sets["TOOTH"]), dtype=np.int64)
        tooth_connectivity = np.stack([element_sets["TOOTH"][int(tag)] for tag in tooth_element_ids])
        np.savez_compressed(
            output_dir / "mesh_data.npz",
            node_ids=np.asarray(sorted(nodes), dtype=np.int64),
            coordinates=np.stack([nodes[tag] for tag in sorted(nodes)]),
            tooth_node_ids=np.asarray(tooth_node_ids, dtype=np.int64),
            indenter_node_ids=np.asarray(indenter_node_ids, dtype=np.int64),
            tooth_element_ids=tooth_element_ids,
            tooth_connectivity=tooth_connectivity,
        )
        gmsh.write(str(output_dir / "case.msh"))
        return deck
    finally:
        gmsh.finalize()


def build_gear_pair_case(case: GearPairCase, output_dir: Path) -> Path:
    """Mesh and write a deformable involute-tooth-pair contact case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("involute_gear_pair_smoke")
        driver_polygon = gear_tooth_polygon(case)
        rotation, translation, contact_point, normal = mating_tooth_transform(case)
        mating_polygon = driver_polygon @ rotation.T + translation

        def add_polygon(points: np.ndarray) -> tuple[int, list[int], list[int]]:
            point_tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in points]
            curve_tags = [
                gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
                for i in range(len(point_tags))
            ]
            loop = gmsh.model.occ.addCurveLoop(curve_tags)
            return gmsh.model.occ.addPlaneSurface([loop]), point_tags, curve_tags

        driver_surface, driver_points, driver_curves = add_polygon(driver_polygon)
        mating_surface, mating_points, mating_curves = add_polygon(mating_polygon)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", case.mesh_size_mm * 0.50)
        gmsh.option.setNumber("Mesh.MeshSizeMax", case.mesh_size_mm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.setSize(
            [(0, tag) for tag in driver_points + mating_points],
            case.mesh_size_mm * 0.70,
        )
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        nodes = {int(tag): coord for tag, coord in zip(node_tags, coordinates)}

        element_sets: dict[str, dict[int, np.ndarray]] = {}
        for name, area in (("DRIVER", driver_surface), ("MATING", mating_surface)):
            types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(2, area)
            elements: dict[int, np.ndarray] = {}
            for element_type, tags, flat_nodes in zip(types, tag_blocks, node_blocks):
                _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if nodes_per_element != 3:
                    continue
                shaped = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 3)
                for row in shaped:
                    a, b, c = (nodes[int(tag)][:2] for tag in row)
                    signed_twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if signed_twice_area < 0:
                        row[1], row[2] = row[2], row[1]
                elements.update({int(tag): row for tag, row in zip(tags, shaped)})
            if not elements:
                raise RuntimeError(f"no linear triangles generated for {name}")
            element_sets[name] = elements

        all_elements = {**element_sets["DRIVER"], **element_sets["MATING"]}
        driver_faces = _surface_edges(all_elements, driver_curves)
        mating_faces = _surface_edges(all_elements, mating_curves)
        if not driver_faces or not mating_faces:
            raise RuntimeError("failed to map mating-tooth contact surfaces")

        driver_node_ids = sorted({int(v) for row in element_sets["DRIVER"].values() for v in row})
        mating_node_ids = sorted({int(v) for row in element_sets["MATING"].values() for v in row})
        root_limit = case.root_radius_mm + case.mesh_size_mm * 0.45
        driver_root = [tag for tag in driver_node_ids if np.linalg.norm(nodes[tag][:2]) <= root_limit]
        mating_root = []
        for tag in mating_node_ids:
            local_xy = rotation.T @ (nodes[tag][:2] - translation)
            if np.linalg.norm(local_xy) <= root_limit:
                mating_root.append(tag)
        if len(driver_root) < 3 or len(mating_root) < 3:
            raise RuntimeError("gear-sector support set is unexpectedly small")

        def ids_block(values: list[int]) -> str:
            return "\n".join(", ".join(map(str, values[i : i + 16])) for i in range(0, len(values), 16))

        lines = ["*HEADING", "Deformable involute tooth-pair contact smoke"]
        lines.append("*NODE")
        lines.extend(f"{tag}, {xyz[0]:.10g}, {xyz[1]:.10g}, {xyz[2]:.10g}" for tag, xyz in sorted(nodes.items()))
        for name, elements in element_sets.items():
            lines.append(f"*ELEMENT, TYPE=CPE3, ELSET={name}")
            lines.extend(f"{tag}, " + ", ".join(map(str, row)) for tag, row in sorted(elements.items()))
        lines.extend(("*NSET, NSET=DRIVERROOT", ids_block(driver_root)))
        lines.extend(("*NSET, NSET=MATINGROOT", ids_block(mating_root)))
        lines.extend(("*NSET, NSET=DRIVERNODES", ids_block(driver_node_ids)))
        lines.extend(("*NSET, NSET=MATINGNODES", ids_block(mating_node_ids)))
        lines.append("*SURFACE, NAME=DRIVERSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in driver_faces)
        lines.append("*SURFACE, NAME=MATINGSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in mating_faces)
        adjust = max(case.initial_gap_mm * 2.0, case.mesh_size_mm * 0.05)
        lines.extend(
            (
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.youngs_modulus_mpa:.10g}, {case.poisson_ratio:.10g}",
                "*SOLID SECTION, ELSET=DRIVER, MATERIAL=STEEL",
                "*SOLID SECTION, ELSET=MATING, MATERIAL=STEEL",
                "*SURFACE INTERACTION, NAME=CONTACTINT",
                "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                f"{case.contact_penalty_factor * case.youngs_modulus_mpa:.10g}",
                "*FRICTION",
                f"{case.friction:.10g}, {0.5 * case.youngs_modulus_mpa:.10g}",
                f"*CONTACT PAIR, INTERACTION=CONTACTINT, TYPE=SURFACE TO SURFACE, ADJUST={adjust:.10g}",
                "MATINGSURF, DRIVERSURF",
                "*BOUNDARY",
                "DRIVERROOT, 1, 2, 0.0",
                "*STEP, NLGEOM",
                "*STATIC",
                "0.025, 1.0, 1e-07, 0.1",
                "*BOUNDARY",
                f"MATINGROOT, 1, 1, {-normal[0] * case.indentation_mm:.10g}",
                f"MATINGROOT, 2, 2, {-normal[1] * case.indentation_mm:.10g}",
                "*NODE FILE",
                "U, RF",
                "*EL FILE",
                "S, E",
                "*CONTACT FILE",
                "CDIS, CSTR, CELS",
                "*END STEP",
            )
        )
        deck = output_dir / "case.inp"
        deck.write_text("\n".join(lines) + "\n", encoding="ascii")
        metadata = {
            **asdict(case),
            "model": "deformable_involute_tooth_pair",
            "contact_point_mm": contact_point.tolist(),
            "contact_normal": normal.tolist(),
            "mating_rotation": rotation.tolist(),
            "mating_center_mm": translation.tolist(),
            "center_distance_mm": float(np.linalg.norm(translation)),
            "nominal_center_distance_mm": 2.0 * case.pitch_radius_mm,
            "nodes": len(nodes),
            "elements": len(all_elements),
            "driver_root_nodes": len(driver_root),
            "mating_root_nodes": len(mating_root),
            "driver_contact_faces": len(driver_faces),
            "mating_contact_faces": len(mating_faces),
        }
        (output_dir / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        driver_element_ids = np.asarray(sorted(element_sets["DRIVER"]), dtype=np.int64)
        np.savez_compressed(
            output_dir / "mesh_data.npz",
            node_ids=np.asarray(sorted(nodes), dtype=np.int64),
            coordinates=np.stack([nodes[tag] for tag in sorted(nodes)]),
            tooth_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            driver_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            mating_node_ids=np.asarray(mating_node_ids, dtype=np.int64),
            driver_element_ids=driver_element_ids,
            driver_connectivity=np.stack([element_sets["DRIVER"][int(tag)] for tag in driver_element_ids]),
            mating_rotation=rotation,
            mating_center=translation,
        )
        gmsh.write(str(output_dir / "case.msh"))
        return deck
    finally:
        gmsh.finalize()


def build_gear_pair_transient_case(case: GearPairTransientCase, output_dir: Path) -> Path:
    """Mesh once at ``phase_start`` and write a chained ``*DYNAMIC`` deck that
    sweeps the mating tooth through ``[phase_start, phase_end]`` with real
    material density and a (possibly time-varying) driver speed profile.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    static_case = case.as_gear_pair_case(case.phase_start)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("involute_gear_pair_transient")
        driver_polygon = gear_tooth_polygon(static_case)
        rotation0, translation0, contact_point, normal0 = mating_tooth_transform(static_case)
        mating_polygon = driver_polygon @ rotation0.T + translation0

        def add_polygon(points: np.ndarray) -> tuple[int, list[int], list[int]]:
            point_tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in points]
            curve_tags = [
                gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
                for i in range(len(point_tags))
            ]
            loop = gmsh.model.occ.addCurveLoop(curve_tags)
            return gmsh.model.occ.addPlaneSurface([loop]), point_tags, curve_tags

        driver_surface, driver_points, driver_curves = add_polygon(driver_polygon)
        mating_surface, mating_points, mating_curves = add_polygon(mating_polygon)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", case.mesh_size_mm * 0.50)
        gmsh.option.setNumber("Mesh.MeshSizeMax", case.mesh_size_mm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.setSize(
            [(0, tag) for tag in driver_points + mating_points], case.mesh_size_mm * 0.70
        )
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        nodes = {int(tag): coord for tag, coord in zip(node_tags, coordinates)}

        element_sets: dict[str, dict[int, np.ndarray]] = {}
        for name, area in (("DRIVER", driver_surface), ("MATING", mating_surface)):
            types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(2, area)
            elements: dict[int, np.ndarray] = {}
            for element_type, tags, flat_nodes in zip(types, tag_blocks, node_blocks):
                _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if nodes_per_element != 3:
                    continue
                shaped = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 3)
                for row in shaped:
                    a, b, c = (nodes[int(tag)][:2] for tag in row)
                    signed_twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                    if signed_twice_area < 0:
                        row[1], row[2] = row[2], row[1]
                elements.update({int(tag): row for tag, row in zip(tags, shaped)})
            if not elements:
                raise RuntimeError(f"no linear triangles generated for {name}")
            element_sets[name] = elements

        all_elements = {**element_sets["DRIVER"], **element_sets["MATING"]}
        driver_faces = _surface_edges(all_elements, driver_curves)
        mating_faces = _surface_edges(all_elements, mating_curves)
        if not driver_faces or not mating_faces:
            raise RuntimeError("failed to map mating-tooth contact surfaces")

        driver_node_ids = sorted({int(v) for row in element_sets["DRIVER"].values() for v in row})
        mating_node_ids = sorted({int(v) for row in element_sets["MATING"].values() for v in row})
        root_limit = case.root_radius_mm + case.mesh_size_mm * 0.45
        driver_root = [tag for tag in driver_node_ids if np.linalg.norm(nodes[tag][:2]) <= root_limit]
        mating_root = []
        for tag in mating_node_ids:
            local_xy = rotation0.T @ (nodes[tag][:2] - translation0)
            if np.linalg.norm(local_xy) <= root_limit:
                mating_root.append(tag)
        if len(driver_root) < 3 or len(mating_root) < 3:
            raise RuntimeError("gear-sector support set is unexpectedly small")

        mating_root_xy0 = {tag: nodes[tag][:2].copy() for tag in mating_root}
        samples = transient_time_samples(
            case.phase_start, case.phase_end, case.n_time_steps, case.mean_speed_rpm,
            case.speed_ripple_fraction,
        )

        def ids_block(values: list[int]) -> str:
            return "\n".join(", ".join(map(str, values[i : i + 16])) for i in range(0, len(values), 16))

        lines = ["*HEADING", "Transient deformable involute tooth-pair contact smoke"]
        lines.append("*NODE")
        lines.extend(f"{tag}, {xyz[0]:.10g}, {xyz[1]:.10g}, {xyz[2]:.10g}" for tag, xyz in sorted(nodes.items()))
        for name, elements in element_sets.items():
            lines.append(f"*ELEMENT, TYPE=CPE3, ELSET={name}")
            lines.extend(f"{tag}, " + ", ".join(map(str, row)) for tag, row in sorted(elements.items()))
        lines.extend(("*NSET, NSET=DRIVERROOT", ids_block(driver_root)))
        lines.extend(("*NSET, NSET=MATINGROOT", ids_block(mating_root)))
        lines.extend(("*NSET, NSET=DRIVERNODES", ids_block(driver_node_ids)))
        lines.extend(("*NSET, NSET=MATINGNODES", ids_block(mating_node_ids)))
        lines.append("*SURFACE, NAME=DRIVERSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in driver_faces)
        lines.append("*SURFACE, NAME=MATINGSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in mating_faces)
        adjust = max(case.initial_gap_mm * 2.0, case.mesh_size_mm * 0.05)
        lines.extend(
            (
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.youngs_modulus_mpa:.10g}, {case.poisson_ratio:.10g}",
                "*DENSITY",
                f"{case.density_tonne_per_mm3:.10g}",
                "*SOLID SECTION, ELSET=DRIVER, MATERIAL=STEEL",
                "*SOLID SECTION, ELSET=MATING, MATERIAL=STEEL",
                "*SURFACE INTERACTION, NAME=CONTACTSTATIC",
                "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                f"{50.0 * case.youngs_modulus_mpa:.10g}",
                "*FRICTION",
                f"{case.friction:.10g}, {0.5 * case.youngs_modulus_mpa:.10g}",
                # A softer penalty specifically for the *DYNAMIC steps.  50x E (tuned
                # for a quasi-rigid *STATIC solve) forms, together with the real
                # tooth-sector mass, an intrinsically stiff spring-mass system with a
                # natural period of order 1e-7 s -- empirically requiring far more
                # Newton increments than fit a smoke-test compute budget even with
                # increment-floor and numerical-damping tuning alone (two prior
                # attempts, see docs/TRANSIENT_FEM_SMOKE_2026-07-16.md).  1x E lowers
                # that penalty by 50x, lengthening the resolvable period roughly
                # sqrt(50) ~ 7x, at the documented cost of more contact interpenetration
                # than the static evidence's 50x E.
                "*SURFACE INTERACTION, NAME=CONTACTDYNAMIC",
                "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                f"{1.0 * case.youngs_modulus_mpa:.10g}",
                "*FRICTION",
                f"{case.friction:.10g}, {0.5 * case.youngs_modulus_mpa:.10g}",
                "*BOUNDARY",
                "DRIVERROOT, 1, 2, 0.0",
            )
        )
        # Step 1 establishes contact and the full indentation load quasi-statically,
        # at the fixed phase_start position, before any real-time dynamics begin,
        # using the proven static-case contact stiffness. Starting *DYNAMIC with
        # contact and load appearing simultaneously in a single microsecond-scale
        # increment was empirically unstable (immediate Newton-Raphson divergence
        # with a ~1e10 residual on iteration 1); a quasi-static preload step gives
        # the dynamic sweep a physically equilibrated, contact-active initial
        # condition instead.
        preload_targets = mating_node_targets(
            case, mating_root_xy0, rotation0, translation0, case.phase_start, load_fraction=1.0
        )
        lines.extend(
            (
                "*STEP, NLGEOM",
                "*STATIC",
                "0.025, 1.0, 1e-07, 0.1",
                f"*CONTACT PAIR, INTERACTION=CONTACTSTATIC, TYPE=SURFACE TO SURFACE, ADJUST={adjust:.10g}",
                "MATINGSURF, DRIVERSURF",
                "*BOUNDARY",
            )
        )
        for node_id, target in sorted(preload_targets.items()):
            lines.append(f"{node_id}, 1, 1, {target[0]:.10g}")
            lines.append(f"{node_id}, 2, 2, {target[1]:.10g}")
        lines.extend(
            (
                "*NODE FILE",
                "U, RF",
                "*EL FILE",
                "S, E",
                "*CONTACT FILE",
                "CDIS, CSTR, CELS",
                "*END STEP",
            )
        )
        # Steps 2..n_time_steps+1 sweep phase_start -> phase_end dynamically, at
        # the constant preload reached above (load_fraction stays 1.0), switching
        # to the softer CONTACTDYNAMIC interaction.
        for k in range(1, len(samples)):
            time_period = float(samples[k, 0] - samples[k - 1, 0])
            phase_k = float(samples[k, 1])
            targets = mating_node_targets(
                case, mating_root_xy0, rotation0, translation0, phase_k, load_fraction=1.0
            )
            lines.extend(
                (
                    "*STEP, NLGEOM",
                    "*DYNAMIC, ALPHA=-0.1",
                    f"{time_period / 16.0:.10g}, {time_period:.10g}, {time_period / 20000.0:.10g}, {time_period / 4.0:.10g}",
                    "*CONTACT PAIR, INTERACTION=CONTACTDYNAMIC, TYPE=SURFACE TO SURFACE",
                    "MATINGSURF, DRIVERSURF",
                    "*BOUNDARY",
                )
            )
            for node_id, target in sorted(targets.items()):
                lines.append(f"{node_id}, 1, 1, {target[0]:.10g}")
                lines.append(f"{node_id}, 2, 2, {target[1]:.10g}")
            lines.extend(
                (
                    "*NODE FILE",
                    "U, RF",
                    "*EL FILE",
                    "S, E",
                    "*CONTACT FILE",
                    "CDIS, CSTR, CELS",
                    "*END STEP",
                )
            )
        deck = output_dir / "case.inp"
        deck.write_text("\n".join(lines) + "\n", encoding="ascii")
        metadata = {
            **asdict(case),
            "model": "transient_deformable_involute_tooth_pair",
            "contact_point_mm": contact_point.tolist(),
            "contact_normal_at_phase_start": normal0.tolist(),
            "mating_rotation_at_phase_start": rotation0.tolist(),
            "mating_center_mm": translation0.tolist(),
            "time_samples_s": samples[:, 0].tolist(),
            "phase_samples": samples[:, 1].tolist(),
            "nodes": len(nodes),
            "elements": len(all_elements),
            "driver_root_nodes": len(driver_root),
            "mating_root_nodes": len(mating_root),
        }
        (output_dir / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        driver_element_ids = np.asarray(sorted(element_sets["DRIVER"]), dtype=np.int64)
        np.savez_compressed(
            output_dir / "mesh_data.npz",
            node_ids=np.asarray(sorted(nodes), dtype=np.int64),
            coordinates=np.stack([nodes[tag] for tag in sorted(nodes)]),
            tooth_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            driver_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            mating_node_ids=np.asarray(mating_node_ids, dtype=np.int64),
            driver_root_ids=np.asarray(driver_root, dtype=np.int64),
            driver_element_ids=driver_element_ids,
            driver_connectivity=np.stack([element_sets["DRIVER"][int(tag)] for tag in driver_element_ids]),
        )
        gmsh.write(str(output_dir / "case.msh"))
        return deck
    finally:
        gmsh.finalize()


def run_calculix(deck: Path, ccx_executable: Path, timeout_seconds: int = 180) -> subprocess.CompletedProcess[str]:
    """Run CalculiX using the deck stem and retain a combined execution log."""
    result = subprocess.run(
        [str(ccx_executable), deck.stem],
        cwd=deck.parent,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    (deck.parent / "solver.log").write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
    return result


def contact_element_counts(log_path: Path) -> list[int]:
    marker = "Number of contact spring elements="
    counts = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker in line:
            counts.append(int(line.split(marker, 1)[1].strip()))
    return counts


def _parse_nodal_blocks(frd_path: Path, marker: str, components: int) -> list[dict[int, np.ndarray]]:
    """Extract every nodal result block matching ``marker`` from an ASCII FRD file, in order."""
    lines = frd_path.read_text(encoding="ascii", errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(marker)]
    if not starts:
        raise ValueError(f"no {marker.strip()} block found in {frd_path}")
    blocks: list[dict[int, np.ndarray]] = []
    for start in starts:
        values: dict[int, np.ndarray] = {}
        active = False
        for line in lines[start + 1 :]:
            if line.startswith(" -1"):
                active = True
                node_id = int(line[3:13])
                tokens = re.findall(r"[-+]?\d\.\d{5}E[-+]\d{3}", line[13:])
                if len(tokens) != components:
                    raise ValueError(f"malformed {marker.strip()} record: {line!r}")
                values[node_id] = np.asarray([float(t) for t in tokens], dtype=np.float32)
            elif active and line.startswith(" -3"):
                break
        if not values:
            raise ValueError(f"empty {marker.strip()} block in {frd_path}")
        blocks.append(values)
    return blocks


def parse_nodal_stress_history(frd_path: Path) -> list[dict[int, np.ndarray]]:
    """Every nodal STRESS block in an ASCII FRD file, in chronological order."""
    return _parse_nodal_blocks(frd_path, " -4  STRESS", 6)


def parse_final_nodal_stress(frd_path: Path) -> dict[int, np.ndarray]:
    """Extract the last nodal STRESS block from a CalculiX ASCII FRD file."""
    return parse_nodal_stress_history(frd_path)[-1]


def parse_nodal_force_history(frd_path: Path) -> list[dict[int, np.ndarray]]:
    """Every nodal FORC (reaction force) block in an ASCII FRD file, in chronological order."""
    return _parse_nodal_blocks(frd_path, " -4  FORC", 3)


def parse_final_nodal_force(frd_path: Path) -> dict[int, np.ndarray]:
    """Extract the last nodal FORC block from a CalculiX ASCII FRD file."""
    return parse_nodal_force_history(frd_path)[-1]


def project_case_to_grid(case_dir: Path, grid_size: int = 64) -> dict[str, np.ndarray]:
    """Interpolate tooth nodal stresses to a fixed grid for operator learning.

    The grid window is derived from the case's own geometry (read from
    ``case.json``), not hardcoded: it reproduces the historical fixed window
    exactly at the default geometry (module=2.0mm, teeth=24,
    root_clearance_module=0.25 -> root_radius=21.5, outer_radius=26 ->
    x in [-3.5, 3.5], y in [19.5, 26.2]), and scales/re-centers for any other
    geometry so the tooth (including its root fillet) stays framed.
    """
    mesh = np.load(case_dir / "mesh_data.npz")
    node_ids = mesh["node_ids"]
    coordinates = mesh["coordinates"]
    tooth_ids = mesh["tooth_node_ids"]
    id_to_index = {int(tag): i for i, tag in enumerate(node_ids)}
    xy = np.stack([coordinates[id_to_index[int(tag)], :2] for tag in tooth_ids])
    stress_map = parse_final_nodal_stress(case_dir / "case.frd")
    stress = np.stack([stress_map[int(tag)] for tag in tooth_ids])[:, [0, 1, 3]]
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    module_mm = float(metadata["module_mm"])
    teeth = float(metadata["teeth"])
    root_clearance_module = float(metadata["root_clearance_module"])
    pitch_radius_mm = 0.5 * module_mm * teeth
    outer_radius_mm = pitch_radius_mm + module_mm
    root_radius_mm = pitch_radius_mm - module_mm * (1.0 + root_clearance_module)
    half_width_mm = 1.75 * module_mm
    margin_below_mm = 1.0 * module_mm
    margin_above_mm = 0.10 * module_mm
    x_axis = np.linspace(-half_width_mm, half_width_mm, grid_size, dtype=np.float32)
    y_axis = np.linspace(
        root_radius_mm - margin_below_mm, outer_radius_mm + margin_above_mm, grid_size, dtype=np.float32
    )
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    query = np.column_stack((xx.ravel(), yy.ravel()))
    mask = griddata(xy, np.ones(len(xy)), query, method="linear", fill_value=0.0).reshape(grid_size, grid_size)
    fields = []
    for component in range(3):
        field = griddata(xy, stress[:, component], query, method="linear", fill_value=0.0)
        fields.append(field.reshape(grid_size, grid_size))
    target = np.stack(fields).astype(np.float32)
    target *= (mask > 0.5)[None]
    return {
        "stress": target,
        "mask": (mask > 0.5).astype(np.float32),
        "x": xx.astype(np.float32),
        "y": yy.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Gear-mesh case with kinematically consistent loading (added 2026-09-27).
#
# ``build_gear_pair_case`` above displaces the mating rim along the line joining
# the closest vertices of the two faceted flank polygons. That direction is not
# the flank normal (it deviates by up to ~90 deg), so the imposed load and the
# direction of friction depend on the flank discretization. The builder below
# keeps the same tooth geometry and meshing but
#   * places the two teeth in conjugate mesh with zero backlash, with the contact
#     point found on densely sampled involutes (so it lies on the line of action),
#   * loads the pair by one rigid translation of the mating gear: ``approach_mm``
#     along the line of action plus a tangential slip with the sign of the
#     kinematic sliding in the running sense (driver clockwise, pushing with its
#     right flank), which reverses at the pitch point,
# so the normal load is set by ``approach_mm`` and the direction of sliding by
# the gear kinematics (it reverses at the pitch point). The driver rim stays fixed;
# the whole motion is expressed in the driver's frame.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GearMeshCase:
    """Quasi-static tooth-pair contact at a given point of the line of action."""

    module_mm: float = 2.0
    teeth: int = 24
    pressure_angle_deg: float = 20.0
    face_width_mm: float = 1.0
    root_clearance_module: float = 0.25
    line_of_action_mm: float = 0.0     # contact position from the pitch point, + toward the driver tip
    approach_mm: float = 0.008         # normal approach imposed along the line of action
    slip_mm: float = 0.015             # imposed tangential slip magnitude (full sliding)
    slip_ramp_mm: float = 0.05         # slip ramps linearly to zero within this distance of the pitch point
    friction: float = 0.08
    youngs_modulus_mpa: float = 210_000.0
    poisson_ratio: float = 0.30
    mesh_size_mm: float = 0.12
    contact_penalty_factor: float = 50.0
    flank_samples: int = 80

    @property
    def pitch_radius_mm(self) -> float:
        return 0.5 * self.module_mm * self.teeth

    @property
    def base_radius_mm(self) -> float:
        return self.pitch_radius_mm * math.cos(math.radians(self.pressure_angle_deg))

    @property
    def outer_radius_mm(self) -> float:
        return self.pitch_radius_mm + self.module_mm

    @property
    def root_radius_mm(self) -> float:
        return self.pitch_radius_mm - self.module_mm * (1.0 + self.root_clearance_module)

    @property
    def single_contact_half_length_mm(self) -> float:
        """Half length of the single-tooth-contact zone on the line of action (equal gears)."""
        rb, ra, rp = self.base_radius_mm, self.outer_radius_mm, self.pitch_radius_mm
        alpha = math.radians(self.pressure_angle_deg)
        path = 2.0 * math.sqrt(ra * ra - rb * rb) - 2.0 * rp * math.sin(alpha)
        base_pitch = math.pi * self.module_mm * math.cos(alpha)
        return base_pitch - 0.5 * path


def _rot(angle: float) -> np.ndarray:
    return np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))), dtype=np.float64)


def gear_mesh_pose(case: GearMeshCase, driver_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and centre of the mating tooth in the driver frame, zero backlash.

    Same kinematics as ``mating_tooth_transform`` (equal gears, the mating gear
    half a tooth pitch out of phase and counter-rotating) without the clearance
    translation.
    """
    tooth_pitch = 2.0 * math.pi / case.teeth
    rotation = _rot(math.pi + 0.5 * tooth_pitch - 2.0 * driver_angle)
    centre = _rot(-driver_angle) @ np.asarray((0.0, 2.0 * case.pitch_radius_mm))
    return rotation, centre


def gear_mesh_contact(case: GearMeshCase, driver_angle: float, samples: int = 4001) -> dict:
    """Contact point, outward flank normal (driver) and line-of-action position.

    Uses densely sampled continuous involutes of the driver's right flank and
    both flanks of the mating tooth.
    """
    from scipy.spatial import cKDTree

    left = conjugate_involute_flank(case, samples)
    right = left.copy()
    right[:, 0] *= -1.0
    rotation, centre = gear_mesh_pose(case, driver_angle)
    mating = np.vstack((left, right)) @ rotation.T + centre
    dist, idx = cKDTree(mating).query(right)
    k = int(np.argmin(dist))
    k = min(max(k, 1), len(right) - 2)
    point = right[k]
    tangent = right[k + 1] - right[k - 1]
    tangent /= np.linalg.norm(tangent)
    if tangent @ point < 0:                        # tangent toward the tooth tip
        tangent = -tangent
    normal = np.array((tangent[1], -tangent[0]))   # outward from the driver's right flank
    if normal[0] < 0:
        normal = -normal
    pitch_point = 0.5 * centre
    # on conjugate involutes the contact lies on the line of action through the pitch point
    s = float(np.sign(np.linalg.norm(point) - case.pitch_radius_mm) * np.linalg.norm(point - pitch_point))
    d_pp = point - pitch_point
    off_line = float(abs(d_pp[0] * normal[1] - d_pp[1] * normal[0]))
    return {"point": point, "normal": normal, "tangent": tangent, "gap": float(dist[k]),
            "line_of_action_mm": s, "off_line_of_action_mm": off_line, "radius_mm": float(np.linalg.norm(point)),
            "rotation": rotation, "centre": centre}


def gear_mesh_angle_for(case: GearMeshCase, target_s: float) -> float:
    """Driver angle (radians) that puts the contact at ``target_s`` on the line of action.

    Conjugate involute action moves the contact along the line of action by r_b per
    radian, so the angle is linear in s. The reference angle is solved once at
    s = -0.8 mm, away from the pitch point where the closest-point search is
    ill-conditioned (the flanks are nearly parallel there).
    """
    s_ref = -0.8
    f = lambda a: gear_mesh_contact(case, a)["line_of_action_mm"] - s_ref
    a0 = -s_ref / case.base_radius_mm
    a1 = a0 + f(a0) / case.base_radius_mm
    for _ in range(20):
        v1 = f(a1)
        if abs(v1) < 1e-9:
            break
        a1 += v1 / case.base_radius_mm
    return float(a1 - (target_s - s_ref) / case.base_radius_mm)


def gear_mesh_contact_exact(case: GearMeshCase, target_s: float) -> dict:
    """Contact point, outward normal and tangent on the driver's right flank for
    contact at ``target_s`` on the line of action, from the involute equations."""
    rb, rp = case.base_radius_mm, case.pitch_radius_mm
    alpha = math.radians(case.pressure_angle_deg)
    half_tooth = math.pi / (2.0 * case.teeth)
    inv = lambda a: math.tan(a) - a

    def flank(r: float) -> np.ndarray:
        theta = 0.5 * math.pi - (half_tooth + inv(alpha) - inv(math.acos(rb / r)))
        return np.array((r * math.cos(theta), r * math.sin(theta)))

    r = math.sqrt(rb * rb + (rp * math.sin(alpha) + target_s) ** 2)
    point = flank(r)
    dr = 1e-5
    tangent = flank(r + dr) - flank(r - dr)
    tangent /= np.linalg.norm(tangent)                 # toward the tooth tip (increasing radius)
    normal = np.array((tangent[1], -tangent[0]))
    if normal[0] < 0:
        normal = -normal
    return {"point": point, "normal": normal, "tangent": tangent, "radius_mm": r, "line_of_action_mm": target_s}


def build_gear_mesh_case(case: GearMeshCase, output_dir: Path) -> Path:
    """Mesh and write a tooth-pair case with kinematically consistent loading.

    The mating tooth is placed in conjugate contact (zero backlash) with the
    contact at ``line_of_action_mm``. In the load step its rim is translated
    rigidly by ``approach_mm`` along the line of action toward the driver, plus
    a tangential slip ``slip_mm`` with the sign of the kinematic sliding for the
    running sense (driver clockwise), ramped to zero at the pitch point. The
    driver rim below the root circle is fixed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    a1 = gear_mesh_angle_for(case, case.line_of_action_mm)
    rot0, centre0 = gear_mesh_pose(case, a1)
    final = gear_mesh_contact_exact(case, case.line_of_action_mm)
    point, normal, tangent = final["point"], final["normal"], final["tangent"]
    # Tangential slip of the mating flank relative to the driver for a rolling
    # increment in the running sense (driver clockwise: a -> a - roll), from the
    # conjugate kinematics: the mating material point at the contact moves from
    # pose(a1 + h) to pose(a1). Its tangential displacement per unit angle is
    # about 2 s (relative rotation about the pitch point) and reverses there.
    h = 1e-5
    rot_h, centre_h = gear_mesh_pose(case, a1 + h)
    xi = rot_h.T @ (point - centre_h)
    slip_rate = float(((rot0 @ xi + centre0) - point) @ tangent) / h
    # One rigid translation of the mating gear: the approach along the line of
    # action (toward the driver) plus a tangential slip with the sign of the
    # kinematic sliding (reversing at the pitch point) and a magnitude that makes
    # the contact slide, ramped to zero within slip_ramp_mm of the pitch point
    # (pure rolling there).
    ramp = min(1.0, abs(case.line_of_action_mm) / case.slip_ramp_mm) if case.slip_ramp_mm > 0 else 1.0
    slip = float(np.sign(slip_rate)) * case.slip_mm * ramp
    approach_vec = -case.approach_mm * normal + slip * tangent
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("involute_gear_mesh")
        driver_polygon = gear_tooth_polygon(case, flank_samples=case.flank_samples)
        mating_polygon = driver_polygon @ rot0.T + centre0

        def add_polygon(points: np.ndarray) -> tuple[int, list[int], list[int]]:
            point_tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in points]
            curve_tags = [
                gmsh.model.occ.addLine(point_tags[i], point_tags[(i + 1) % len(point_tags)])
                for i in range(len(point_tags))
            ]
            loop = gmsh.model.occ.addCurveLoop(curve_tags)
            return gmsh.model.occ.addPlaneSurface([loop]), point_tags, curve_tags

        driver_surface, driver_points, driver_curves = add_polygon(driver_polygon)
        mating_surface, mating_points, mating_curves = add_polygon(mating_polygon)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMin", case.mesh_size_mm * 0.50)
        gmsh.option.setNumber("Mesh.MeshSizeMax", case.mesh_size_mm)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.setSize([(0, tag) for tag in driver_points + mating_points], case.mesh_size_mm * 0.70)
        gmsh.model.mesh.generate(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        nodes = {int(tag): coord for tag, coord in zip(node_tags, coordinates)}
        element_sets: dict[str, dict[int, np.ndarray]] = {}
        for name, area in (("DRIVER", driver_surface), ("MATING", mating_surface)):
            types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(2, area)
            elements: dict[int, np.ndarray] = {}
            for element_type, tags, flat_nodes in zip(types, tag_blocks, node_blocks):
                _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
                if nodes_per_element != 3:
                    continue
                shaped = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 3)
                for row in shaped:
                    a, b, c = (nodes[int(tag)][:2] for tag in row)
                    if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                        row[1], row[2] = row[2], row[1]
                elements.update({int(tag): row for tag, row in zip(tags, shaped)})
            if not elements:
                raise RuntimeError(f"no linear triangles generated for {name}")
            element_sets[name] = elements

        all_elements = {**element_sets["DRIVER"], **element_sets["MATING"]}
        driver_faces = _surface_edges(all_elements, driver_curves)
        mating_faces = _surface_edges(all_elements, mating_curves)
        if not driver_faces or not mating_faces:
            raise RuntimeError("failed to map contact surfaces")
        driver_node_ids = sorted({int(v) for row in element_sets["DRIVER"].values() for v in row})
        mating_node_ids = sorted({int(v) for row in element_sets["MATING"].values() for v in row})
        root_limit = case.root_radius_mm + case.mesh_size_mm * 0.45
        driver_root = [t for t in driver_node_ids if np.linalg.norm(nodes[t][:2]) <= root_limit]
        mating_root = [t for t in mating_node_ids if np.linalg.norm(rot0.T @ (nodes[t][:2] - centre0)) <= root_limit]
        if len(driver_root) < 3 or len(mating_root) < 3:
            raise RuntimeError("gear-sector support set is unexpectedly small")

        def ids_block(values: list[int]) -> str:
            return "\n".join(", ".join(map(str, values[i : i + 16])) for i in range(0, len(values), 16))

        # prescribed rigid translation of the mating rim (approach plus kinematic slip)
        mating_bc = []
        for t in mating_root:
            u = approach_vec
            mating_bc += [f"{t}, 1, 1, {u[0]:.12g}", f"{t}, 2, 2, {u[1]:.12g}"]

        lines = ["*HEADING", "Involute tooth-pair contact, kinematically consistent loading"]
        lines.append("*NODE")
        lines.extend(f"{tag}, {xyz[0]:.10g}, {xyz[1]:.10g}, {xyz[2]:.10g}" for tag, xyz in sorted(nodes.items()))
        for name, elements in element_sets.items():
            lines.append(f"*ELEMENT, TYPE=CPE3, ELSET={name}")
            lines.extend(f"{tag}, " + ", ".join(map(str, row)) for tag, row in sorted(elements.items()))
        lines.extend(("*NSET, NSET=DRIVERROOT", ids_block(driver_root)))
        lines.extend(("*NSET, NSET=MATINGROOT", ids_block(mating_root)))
        lines.extend(("*NSET, NSET=DRIVERNODES", ids_block(driver_node_ids)))
        lines.extend(("*NSET, NSET=MATINGNODES", ids_block(mating_node_ids)))
        lines.append("*SURFACE, NAME=DRIVERSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in driver_faces)
        lines.append("*SURFACE, NAME=MATINGSURF, TYPE=ELEMENT")
        lines.extend(f"{element}, {face}" for element, face in mating_faces)
        lines.extend(
            (
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.youngs_modulus_mpa:.10g}, {case.poisson_ratio:.10g}",
                "*SOLID SECTION, ELSET=DRIVER, MATERIAL=STEEL",
                "*SOLID SECTION, ELSET=MATING, MATERIAL=STEEL",
                "*SURFACE INTERACTION, NAME=CONTACTINT",
                "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR",
                f"{case.contact_penalty_factor * case.youngs_modulus_mpa:.10g}",
                "*FRICTION",
                f"{case.friction:.10g}, {0.5 * case.youngs_modulus_mpa:.10g}",
                # No ADJUST: the conjugate placement already puts the flanks in point
                # contact, and ADJUST would close the curvature gap x^2/(2R') over
                # every slave node within its distance (about +-0.2 mm for 6 um),
                # turning the Hertzian contact into a mesh-dependent flat punch.
                "*CONTACT PAIR, INTERACTION=CONTACTINT, TYPE=SURFACE TO SURFACE",
                "MATINGSURF, DRIVERSURF",
                "*BOUNDARY",
                "DRIVERROOT, 1, 2, 0.0",
                "*STEP, NLGEOM",
                "*STATIC",
                "0.025, 1.0, 1e-07, 0.1",
                "*BOUNDARY",
                *mating_bc,
                "*NODE FILE",
                "U, RF",
                "*EL FILE",
                "S, E",
                "*CONTACT FILE",
                "CDIS, CSTR, CELS",
                "*END STEP",
            )
        )
        deck = output_dir / "case.inp"
        deck.write_text("\n".join(lines) + "\n", encoding="ascii")
        metadata = {
            **asdict(case),
            "model": "involute_tooth_pair_kinematic_loading",
            "contact_point_mm": point.tolist(),
            "contact_normal": normal.tolist(),
            "contact_tangent": final["tangent"].tolist(),
            "contact_radius_mm": final["radius_mm"],
            "driver_angle_rad": a1,
            "slip_per_rolling_angle_mm": slip_rate,
            "imposed_slip_toward_tip_mm": slip,
            "approach_vector_mm": approach_vec.tolist(),
            "mating_rotation": rot0.tolist(),
            "mating_center_mm": centre0.tolist(),

            "nodes": len(nodes),
            "elements": len(all_elements),
            "driver_root_nodes": len(driver_root),
            "mating_root_nodes": len(mating_root),
        }
        (output_dir / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        driver_element_ids = np.asarray(sorted(element_sets["DRIVER"]), dtype=np.int64)
        np.savez_compressed(
            output_dir / "mesh_data.npz",
            node_ids=np.asarray(sorted(nodes), dtype=np.int64),
            coordinates=np.stack([nodes[tag] for tag in sorted(nodes)]),
            tooth_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            driver_node_ids=np.asarray(driver_node_ids, dtype=np.int64),
            mating_node_ids=np.asarray(mating_node_ids, dtype=np.int64),
            driver_element_ids=driver_element_ids,
            driver_connectivity=np.stack([element_sets["DRIVER"][int(tag)] for tag in driver_element_ids]),
            mating_rotation=rot0,
            mating_center=centre0,
        )
        gmsh.write(str(output_dir / "case.msh"))
        return deck
    finally:
        gmsh.finalize()
