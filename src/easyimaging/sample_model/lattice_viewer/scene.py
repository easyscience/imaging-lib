# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""The things to draw for a lattice: atoms, bonds, the unit cell and the lattice vectors."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .elements import ELEMENT_STYLES

Coords = Sequence[float, float, float]

if TYPE_CHECKING:
    from ..lattice import Lattice

# Atoms are drawn at half their covalent radius, so the inside of the unit cell stays visible
ATOM_RADIUS_SCALE = 0.5

# Fractional coordinates closer than this to 0 are considered to lie on the cell boundary
BOUNDARY_TOLERANCE = 1e-6

# Two atoms are bonded if they are no further apart than the sum of their covalent radii plus a tolerance in Å,
# and no further apart than a factor times the distance from either atom to its nearest neighbour. The second
# rule keeps only the first coordination shell, so that e.g. rutile TiO2 gets no Ti-Ti bonds.
BOND_LENGTH_TOLERANCE = 0.25
COORDINATION_SHELL_FACTOR = 1.3
BOND_RADIUS = 0.08

CELL_EDGE_COLOR = '#555555'

LATTICE_VECTOR_OVERHANG = 0.3
LATTICE_VECTOR_COLORS = {'a': '#DC2828', 'b': '#28B428', 'c': '#2850DC'}
ARROW_SHAFT_RADIUS = 0.05
ARROW_HEAD_RADIUS = 0.15
ARROW_HEAD_LENGTH = 0.45
ARROW_LABEL_DISTANCE = 1.05  # Fractional of the vector


@dataclass
class AtomSphere:
    """A sphere representing one atom."""

    element: str
    label: str  # The atomic species, e.g. 'Fe56'
    site_name: str  # The display name of the atom site
    position: Coords  # Cartesian coordinates in Å
    color: str  # Hex string
    radius: float  # In Å


@dataclass
class BondCylinder:
    """A cylinder representing half of a bond, coloured like the atom at its start."""

    start: Coords  # Cartesian coordinates in Å
    end: Coords  # Cartesian coordinates in Å
    color: str  # Hex string
    radius: float  # In Å


@dataclass
class LatticeVectorArrow:
    """An arrow representing one lattice vector, with its label (``'a'``, ``'b'`` or ``'c'``) beyond the tip."""

    label: str
    start: Coords  # Cartesian coordinates in Å
    end: Coords  # Cartesian coordinates in Å
    label_position: Coords  # Cartesian coordinates in Å
    color: str  # Hex string
    shaft_radius: float  # In Å
    head_radius: float  # In Å
    head_length: float  # In Å


def build_scene(lattice: Lattice) -> dict:
    """Return everything to draw for a lattice, ready to be converted to JSON for the web page that draws it.

    Parameters
    ----------
    lattice : Lattice
        The lattice to draw.

    Returns
    -------
    dict
        The atoms, bonds, cell edges and lattice vectors to draw.
    """
    spheres = atom_spheres(lattice)
    return {
        'atoms': [asdict(sphere) for sphere in spheres],
        'bonds': [asdict(cylinder) for cylinder in bond_cylinders(spheres)],
        'cell_edges': cell_edges(lattice),
        'cell_edge_color': CELL_EDGE_COLOR,
        'lattice_vectors': [asdict(arrow) for arrow in lattice_vector_arrows(lattice)],
    }


def atom_spheres(lattice: Lattice) -> list[AtomSphere]:
    """Return one sphere per atom in the unit cell.

    An atom on a face, edge or corner of the unit cell is also drawn on the opposite faces, edges or corners,
    so the cell looks complete.

    Parameters
    ----------
    lattice : Lattice
        The lattice to draw.

    Returns
    -------
    list[AtomSphere]
        The spheres to draw.
    """
    spheres = []
    for site in lattice.atom_sites:
        element = site.atom.element
        style = ELEMENT_STYLES[element]
        for fractional_position in _fractional_positions_to_draw(site.fract_a.value, site.fract_b.value, site.fract_c.value):
            sphere = AtomSphere(
                element=element,
                label=site.atom.name,
                site_name=site.display_name,
                position=_to_cartesian(fractional_position, lattice),
                color=style.color,
                radius=ATOM_RADIUS_SCALE * style.covalent_radius,
            )
            spheres.append(sphere)
    return spheres


def bond_cylinders(spheres: list[AtomSphere]) -> list[BondCylinder]:
    """Return the bonds between the atoms drawn in the unit cell.

    Each bond is drawn as two cylinders that meet in the middle, each coloured like the atom it starts from.

    Parameters
    ----------
    spheres : list[AtomSphere]
        The atoms drawn in the unit cell, from [`atom_spheres`][..atom_spheres].

    Returns
    -------
    list[BondCylinder]
        The cylinders to draw, two per bond.
    """
    nearest_neighbour_distances = []
    for sphere in spheres:
        distances_to_others = [math.dist(sphere.position, other.position) for other in spheres if other is not sphere]
        nearest_neighbour_distances.append(min(distances_to_others, default=math.inf))

    cylinders = []
    for i, sphere_1 in enumerate(spheres):
        for j, sphere_2 in enumerate(spheres[i + 1 :]):
            distance = math.dist(sphere_1.position, sphere_2.position)
            covalent_cutoff = (
                ELEMENT_STYLES[sphere_1.element].covalent_radius
                + ELEMENT_STYLES[sphere_2.element].covalent_radius
                + BOND_LENGTH_TOLERANCE
            )
            shortest_nearest_neighbour_distance = min(nearest_neighbour_distances[i], nearest_neighbour_distances[i + j + 1])
            coordination_shell_cutoff = COORDINATION_SHELL_FACTOR * shortest_nearest_neighbour_distance
            if distance <= min(covalent_cutoff, coordination_shell_cutoff):
                middle = ((np.array(sphere_1.position) + np.array(sphere_2.position)) / 2).tolist()
                cylinders.append(BondCylinder(start=sphere_1.position, end=middle, color=sphere_1.color, radius=BOND_RADIUS))
                cylinders.append(BondCylinder(start=sphere_2.position, end=middle, color=sphere_2.color, radius=BOND_RADIUS))
    return cylinders


def cell_edges(lattice: Lattice) -> list[tuple[Coords, Coords]]:
    """Return the 12 edges of the unit cell.

    Parameters
    ----------
    lattice : Lattice
        The lattice to draw.

    Returns
    -------
    list[tuple[list[float, float, float], list[float, float, float]]]
        The start and end point of each edge, in Cartesian coordinates in Å.
    """
    vectors = np.array(lattice.lattice_vectors)
    # Add the edges from the origin.
    edges = [([0.0, 0.0, 0.0], vector.tolist()) for vector in vectors]
    # Remaining edges are obtained by shifting the origin by any two lattice vectors and then add the connecting edges.
    opposite_corner = np.sum(vectors, axis=0)
    for i, vector in enumerate(vectors):
        shifted_origin = opposite_corner - vector
        vector_shifts = vectors.copy()
        vector_shifts[i] *= -1  # Flip the i-th vector since it was subtracted from the origin.
        # Edges from this origin are defined by subtracting the vector shifts.
        for shift in vector_shifts:
            edges.append((shifted_origin.tolist(), (shifted_origin - shift).tolist()))
    return edges


def lattice_vector_arrows(lattice: Lattice) -> list[LatticeVectorArrow]:
    """Return the arrows for the lattice vectors a, b and c, starting at the origin.

    Parameters
    ----------
    lattice : Lattice
        The lattice to draw.

    Returns
    -------
    list[LatticeVectorArrow]
        The arrows to draw.
    """
    vectors = np.array(lattice.lattice_vectors)
    vector_lengths = np.array([lattice.length_a.value, lattice.length_b.value, lattice.length_c.value])
    # Overhang depends on the longest vector so that all lattice vectors are always clearly visible regardless of very askew
    # lattices.
    overhang = LATTICE_VECTOR_OVERHANG * vector_lengths.max()
    arrows = []
    for label, vector, length in zip('abc', vectors, vector_lengths):
        direction = vector / length
        end = vector + overhang * direction
        arrow = LatticeVectorArrow(
            label=label,
            start=[0.0, 0.0, 0.0],
            end=end.tolist(),
            label_position=(end * ARROW_LABEL_DISTANCE).tolist(),
            color=LATTICE_VECTOR_COLORS[label],
            shaft_radius=ARROW_SHAFT_RADIUS,
            head_radius=ARROW_HEAD_RADIUS,
            head_length=ARROW_HEAD_LENGTH,
        )
        arrows.append(arrow)
    return arrows


def _fractional_positions_to_draw(fract_a: float, fract_b: float, fract_c: float) -> list[Coords]:
    """Return the fractional position of an atom, plus its copies on the opposite side of the cell for each
    coordinate that lies on the cell boundary.

    Fractional coordinates are always in the range [0, 1), so only a coordinate of 0 needs a copy at 1.
    """
    values_per_axis = [[0.0, 1.0] if abs(value) < BOUNDARY_TOLERANCE else [value] for value in (fract_a, fract_b, fract_c)]
    A, B, C = values_per_axis
    return [(a, b, c) for a in A for b in B for c in C]


def _to_cartesian(fractional_position: Coords, lattice: Lattice) -> Coords:
    """Convert a position in fractional coordinates to Cartesian coordinates in Å."""
    return (np.array(fractional_position) @ np.array(lattice.lattice_vectors)).tolist()
