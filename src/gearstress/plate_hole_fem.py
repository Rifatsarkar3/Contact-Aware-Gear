"""CalculiX/Gmsh backend for a plate-with-elliptical-hole elasticity problem.

Independent second domain from the gear-tooth-contact study: a rectangular
linear-elastic plate in uniaxial tension with a central elliptical hole, no
contact, no friction, no gear geometry at all. Built to test whether the
paper's central small-sample architecture-misranking finding generalizes
beyond gear-tooth contact to an unrelated PDE-surrogate problem.

Validated against Inglis's (1913) closed-form elliptical-hole stress
concentration factor Kt = 1 + 2*(a/b), where `a` is the hole's semi-axis
perpendicular to the applied tension (along x here) and `b` is its semi-axis
parallel to it (along y, the loading direction). a=b reduces to Kirsch's
classical circular-hole result, Kt=3. This closed-form solution is
independent of Young's modulus and Poisson's ratio for a traction-free hole
boundary in a plate large relative to the hole (St. Venant's principle),
which is why E and nu are held fixed across every generated case -- varying
them would not add real variability to the stress field, only to
displacements, unlike the gear-contact problem where E genuinely changes
Hertzian contact stiffness.

Geometry: plate ``plate_width_mm`` x ``plate_height_mm`` (in the x/y plane,
centered at the origin), y being the load axis. Bottom edge (y=-H/2) is
pinned against vertical motion (with one node also pinned horizontally, to
remove rigid-body drift); the top edge (y=+H/2) is displaced upward by a
prescribed amount, both edges otherwise free -- a standard "grips far from
the feature of interest" tension setup. Plane strain (CPE3), matching the
gear study's convention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import gmsh
import numpy as np


@dataclass(frozen=True)
class PlateHoleCase:
    a_mm: float = 3.0
    b_mm: float = 3.0
    plate_width_mm: float = 150.0
    plate_height_mm: float = 150.0
    far_field_stress_mpa: float = 100.0
    youngs_modulus_mpa: float = 207_000.0
    poisson_ratio: float = 0.30
    mesh_size_mm: float = 6.0
    hole_mesh_size_mm: float = 0.4

    @property
    def aspect_ratio(self) -> float:
        return self.a_mm / self.b_mm

    @property
    def kt_theory(self) -> float:
        """Inglis (1913) elliptical-hole stress concentration factor."""
        return 1.0 + 2.0 * self.aspect_ratio

    @property
    def top_displacement_mm(self) -> float:
        """Rough displacement estimate to land near the target far-field stress.

        Only needs to be approximately right: the actual far-field stress is
        measured directly from the solved field (see
        ``measure_stress_concentration``), not assumed from this formula.
        """
        strain = self.far_field_stress_mpa / self.youngs_modulus_mpa
        return strain * self.plate_height_mm / 2.0


def build_plate_hole_case(case: PlateHoleCase, output_dir: Path) -> Path:
    """Mesh a plate-with-elliptical-hole case and write a CalculiX input deck."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("plate_hole")
        w, h = case.plate_width_mm, case.plate_height_mm
        plate = gmsh.model.occ.addRectangle(-w / 2.0, -h / 2.0, 0.0, w, h)
        hole = gmsh.model.occ.addDisk(0.0, 0.0, 0.0, case.a_mm, case.b_mm)
        gmsh.model.occ.synchronize()
        cut, _ = gmsh.model.occ.cut([(2, plate)], [(2, hole)])
        gmsh.model.occ.synchronize()
        plate_surface = cut[0][1]

        hole_boundary = [
            tag
            for dim, tag in gmsh.model.getBoundary([(2, plate_surface)], oriented=False, combined=False)
            if dim == 1
            and abs(gmsh.model.occ.getCenterOfMass(1, tag)[0]) < w / 4.0
            and abs(gmsh.model.occ.getCenterOfMass(1, tag)[1]) < h / 4.0
        ]
        hole_points = [
            (dim, tag)
            for dim, tag in gmsh.model.getBoundary(
                [(1, t) for t in hole_boundary], oriented=False, recursive=True
            )
            if dim == 0
        ]

        # Second-order (6-node) triangles: CalculiX assigns one constant stress
        # value per *linear* (CPE3) element to all of its nodes, which chronically
        # underestimates a sharp stress concentration at a curved traction-free
        # boundary like this hole -- confirmed empirically (a first-order mesh
        # here under-predicted the classical Kt by ~47%, not a measurement bug:
        # both the interpolated far-field stress and an independent global
        # force-balance check agreed). Quadratic elements capture the true
        # within-element stress gradient instead of one flat value per element.
        #
        # A distance field (not just a fine size at the boundary points) controls
        # radial mesh growth away from the hole: point-based sizing alone left
        # the transition to the coarse far-field mesh too abrupt, which both
        # under-resolved the true peak stress and, for sharper ellipses, produced
        # badly-shaped quadratic elements (nonpositive Jacobian) where the fine
        # and coarse regions met.
        # Fine element size scales with the hole's own sharpest local curvature
        # radius (b^2/a, minimal at the high-stress tip) rather than a fixed
        # constant -- a fixed size that resolves a circular hole is far too
        # coarse for an elongated ellipse's much tighter tip curvature.
        min_curvature_radius = (case.b_mm ** 2) / case.a_mm
        fine_size = min(case.hole_mesh_size_mm, 0.06 * min_curvature_radius)
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [float(t) for t in hole_boundary])
        gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 300)
        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", fine_size)
        gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", case.mesh_size_mm)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.5 * min_curvature_radius)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", 8.0 * case.a_mm)
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates).reshape(-1, 3)
        nodes = {int(tag): coord for tag, coord in zip(node_tags, coordinates)}

        elements: dict[int, np.ndarray] = {}
        types, tag_blocks, node_blocks = gmsh.model.mesh.getElements(2, plate_surface)
        for element_type, tags, flat_nodes in zip(types, tag_blocks, node_blocks):
            _, _, _, nodes_per_element, _, _ = gmsh.model.mesh.getElementProperties(element_type)
            if nodes_per_element != 6:
                continue
            shaped = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 6)
            for row in shaped:
                a, b, c = (nodes[int(tag)][:2] for tag in row[:3])
                signed_twice_area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                if signed_twice_area <= 0:
                    raise RuntimeError(f"inverted or degenerate element {row.tolist()}, area={signed_twice_area}")
            elements.update({int(tag): row for tag, row in zip(tags, shaped)})
        if not elements:
            raise RuntimeError("no quadratic (6-node) triangles generated for plate surface")

        tol = case.mesh_size_mm * 0.1
        bottom_nodes = sorted(tag for tag, xyz in nodes.items() if abs(xyz[1] - (-h / 2.0)) < tol)
        top_nodes = sorted(tag for tag, xyz in nodes.items() if abs(xyz[1] - (h / 2.0)) < tol)
        if len(bottom_nodes) < 2 or len(top_nodes) < 2:
            raise RuntimeError(f"expected grip edges, found bottom={len(bottom_nodes)} top={len(top_nodes)}")
        pin_node = min(bottom_nodes, key=lambda tag: nodes[tag][0])

        def ids_block(values: list[int]) -> str:
            return "\n".join(", ".join(map(str, values[i : i + 16])) for i in range(0, len(values), 16))

        lines = ["*HEADING", "Paper 01 second-domain: plate with elliptical hole, uniaxial tension"]
        lines.append("*NODE")
        lines.extend(f"{tag}, {xyz[0]:.10g}, {xyz[1]:.10g}, {xyz[2]:.10g}" for tag, xyz in sorted(nodes.items()))
        lines.append("*ELEMENT, TYPE=CPE6, ELSET=PLATE")
        lines.extend(f"{tag}, " + ", ".join(map(str, row)) for tag, row in sorted(elements.items()))
        lines.extend(("*NSET, NSET=BOTTOM", ids_block(bottom_nodes)))
        lines.extend(("*NSET, NSET=TOP", ids_block(top_nodes)))
        lines.extend(("*NSET, NSET=PINNODE", ids_block([pin_node])))
        lines.extend(
            (
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.youngs_modulus_mpa:.10g}, {case.poisson_ratio:.10g}",
                "*SOLID SECTION, ELSET=PLATE, MATERIAL=STEEL",
                "*BOUNDARY",
                "BOTTOM, 2, 2, 0.0",
                "PINNODE, 1, 1, 0.0",
                "*STEP",
                "*STATIC",
                "*BOUNDARY",
                f"TOP, 2, 2, {case.top_displacement_mm:.10g}",
                "*NODE FILE",
                "U, RF",
                "*EL FILE",
                "S, E",
                "*END STEP",
            )
        )
        deck = output_dir / "case.inp"
        deck.write_text("\n".join(lines) + "\n", encoding="ascii")
        metadata = {**asdict(case), "aspect_ratio": case.aspect_ratio, "kt_theory": case.kt_theory, "nodes": len(nodes), "elements": len(elements)}
        (output_dir / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(
            output_dir / "mesh_data.npz",
            node_ids=np.asarray(sorted(nodes), dtype=np.int64),
            coordinates=np.stack([nodes[tag] for tag in sorted(nodes)]),
        )
        return deck
    finally:
        gmsh.finalize()


def measure_stress_concentration(case_dir: Path) -> dict[str, float]:
    """Measure Kt = (peak sigma_yy near the hole boundary) / (far-field sigma_yy).

    The far-field reference is interpolated (scipy griddata, linear) at an
    exact point well away from the hole, on the same y=0 plane it sits in --
    the classical infinite-plate problem's own reference convention; the
    coarse, unstructured mesh out there makes interpolation the right tool.
    The peak, by contrast, is read directly from the closest *actual* mesh
    node to the theoretical peak location (+-a, 0) on the hole boundary
    (mirrored, since either tip is a valid peak by symmetry), not
    interpolated: the near-tip gradient is steep enough that linearly
    interpolating between scattered points slightly off the boundary
    silently underestimated the true peak by 20-30% in early testing, even
    though a mesh node already sat almost exactly on the boundary with the
    correct value. Direct-node lookup is only valid because the boundary
    mesh is finely and deliberately refined there (see ``fine_size`` in
    ``build_plate_hole_case``) -- it would not be a safe substitute for
    interpolation anywhere the mesh is coarse.
    """
    from scipy.interpolate import griddata

    from gearstress.fem import parse_final_nodal_stress

    mesh = np.load(case_dir / "mesh_data.npz")
    node_ids = mesh["node_ids"]
    coordinates = mesh["coordinates"]
    metadata = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    a_mm, w_mm = float(metadata["a_mm"]), float(metadata["plate_width_mm"])

    stress_map = parse_final_nodal_stress(case_dir / "case.frd")
    xy = coordinates[:, :2]
    syy = np.asarray([stress_map[int(tag)][1] for tag in node_ids])

    dist_to_right_tip = np.hypot(xy[:, 0] - a_mm, xy[:, 1])
    dist_to_left_tip = np.hypot(xy[:, 0] + a_mm, xy[:, 1])
    right_idx = int(dist_to_right_tip.argmin())
    left_idx = int(dist_to_left_tip.argmin())
    closest_gap = min(dist_to_right_tip[right_idx], dist_to_left_tip[left_idx])
    if closest_gap > 0.05 * max(a_mm, 1.0):
        raise RuntimeError(
            f"no mesh node close enough to the theoretical peak location (closest is "
            f"{closest_gap:.4f}mm away) -- boundary mesh is not fine enough to trust direct lookup"
        )
    peak_syy = float(max(syy[right_idx], syy[left_idx]))

    farfield_point = np.array([[0.4 * w_mm, 0.0]])
    farfield_syy = griddata(xy, syy, farfield_point, method="linear")[0]
    if np.isnan(farfield_syy):
        farfield_syy = griddata(xy, syy, farfield_point, method="nearest")[0]
    farfield_syy = float(farfield_syy)

    return {
        "peak_syy_mpa": peak_syy,
        "farfield_syy_mpa": farfield_syy,
        "kt_fem": peak_syy / farfield_syy,
        "kt_theory": float(metadata["kt_theory"]),
    }


def project_plate_hole_case_to_grid(case_dir: Path, grid_size: int = 64, window_mm: float = 20.0) -> dict[str, np.ndarray]:
    """Interpolate nodal stresses onto a fixed square grid, centered on the hole.

    A fixed window (default +-20mm) is used across every case regardless of
    the actual hole size, matching the gear pipeline's fixed-grid convention
    -- the operator learns on a consistent grid, not one that rescales with
    the input geometry.
    """
    from scipy.interpolate import griddata

    from gearstress.fem import parse_final_nodal_stress

    mesh = np.load(case_dir / "mesh_data.npz")
    node_ids = mesh["node_ids"]
    xy = mesh["coordinates"][:, :2]
    stress_map = parse_final_nodal_stress(case_dir / "case.frd")
    stress = np.stack([stress_map[int(tag)][[0, 1, 3]] for tag in node_ids])

    axis = np.linspace(-window_mm, window_mm, grid_size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
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
