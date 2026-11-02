# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import json
import math

import numpy as np

from easyimaging.sample_model import Atoms
from easyimaging.sample_model import AtomSite
from easyimaging.sample_model import Lattice
from easyimaging.sample_model import crystals
from easyimaging.sample_model._lattice_viewer.elements import ELEMENT_STYLES
from easyimaging.sample_model._lattice_viewer.scene import ATOM_RADIUS_SCALE
from easyimaging.sample_model._lattice_viewer.scene import LATTICE_VECTOR_OVERHANG
from easyimaging.sample_model._lattice_viewer.scene import atom_spheres
from easyimaging.sample_model._lattice_viewer.scene import bond_cylinders
from easyimaging.sample_model._lattice_viewer.scene import build_scene
from easyimaging.sample_model._lattice_viewer.scene import cell_edges
from easyimaging.sample_model._lattice_viewer.scene import lattice_vector_arrows


def test_every_element_has_a_style():
    # When Then Expect
    for atom in Atoms:
        assert atom.element in ELEMENT_STYLES


def test_atom_spheres_body_centered_cubic():
    # When
    lattice = crystals.body_centered_cubic(length_a=2.0, atom='Fe', debye_temperature=300)
    # Then
    spheres = atom_spheres(lattice)
    # Expect
    corners = [[x, y, z] for x in (0.0, 2.0) for y in (0.0, 2.0) for z in (0.0, 2.0)]
    expected_positions = corners + [[1.0, 1.0, 1.0]]
    assert len(spheres) == 9  # 8 corner atoms and 1 body-centre atom
    assert sorted(np.round(sphere.position, 6).tolist() for sphere in spheres) == sorted(expected_positions)
    for sphere in spheres:
        assert sphere.element == 'Fe'
        assert sphere.label == 'Fe'
        assert sphere.color == ELEMENT_STYLES['Fe'].color
        assert sphere.radius == ATOM_RADIUS_SCALE * ELEMENT_STYLES['Fe'].covalent_radius
    assert {sphere.site_name for sphere in spheres} == {site.display_name for site in lattice.atom_sites}


def test_atom_spheres_label_includes_isotope():
    # When
    site = AtomSite(atom='Fe56', fract_a=0.5, fract_b=0.5, fract_c=0.5, debye_temperature=300, display_name='Fe1')
    lattice = Lattice(length_a=2.0, length_b=2.0, length_c=2.0, atom_sites=[site])
    # Then
    (sphere,) = atom_spheres(lattice)
    # Expect
    assert sphere.element == 'Fe'
    assert sphere.label == 'Fe56'
    assert sphere.site_name == 'Fe1'


def test_atom_spheres_atom_on_face_is_drawn_on_both_faces():
    # When
    site = AtomSite(atom='O', fract_a=0.5, fract_b=0.5, fract_c=0.0, debye_temperature=300)
    lattice = Lattice(length_a=2.0, length_b=2.0, length_c=4.0, atom_sites=[site])
    # Then
    spheres = atom_spheres(lattice)
    # Expect
    assert sorted(np.round(sphere.position, 6).tolist() for sphere in spheres) == [[1.0, 1.0, 0.0], [1.0, 1.0, 4.0]]


def test_cell_edges_have_lattice_lengths():
    # When
    lattice = Lattice(length_a=1.0, length_b=2.0, length_c=3.0, alpha=80.0, beta=85.0, gamma=95.0)
    # Then
    edges = cell_edges(lattice)
    # Expect
    edge_lengths = sorted(round(float(np.linalg.norm(np.subtract(end, start))), 6) for start, end in edges)
    assert edge_lengths == [1.0] * 4 + [2.0] * 4 + [3.0] * 4


def test_bond_cylinders_zincblende():
    # When
    lattice = crystals.zincblende(length_a=5.43, atom1='Zn', atom2='S', debye_temperature1=300, debye_temperature2=300)
    # Then
    cylinders = bond_cylinders(atom_spheres(lattice))
    # Expect
    # Each of the 4 S atoms inside the cell is bonded to 4 Zn atoms; each bond is drawn as 2 half bonds
    assert len(cylinders) == 2 * 16
    zn_s_bond_length = 5.43 * math.sqrt(3) / 4
    for cylinder in cylinders:
        assert math.isclose(2 * math.dist(cylinder.start, cylinder.end), zn_s_bond_length)


def test_bond_cylinders_rutile_has_no_ti_ti_bonds():
    # When
    x = 0.305
    atom_sites = [
        AtomSite(atom='Ti', fract_a=0, fract_b=0, fract_c=0, debye_temperature=300),
        AtomSite(atom='Ti', fract_a=0.5, fract_b=0.5, fract_c=0.5, debye_temperature=300),
        AtomSite(atom='O', fract_a=x, fract_b=x, fract_c=0, debye_temperature=300),
        AtomSite(atom='O', fract_a=1 - x, fract_b=1 - x, fract_c=0, debye_temperature=300),
        AtomSite(atom='O', fract_a=0.5 + x, fract_b=0.5 - x, fract_c=0.5, debye_temperature=300),
        AtomSite(atom='O', fract_a=0.5 - x, fract_b=0.5 + x, fract_c=0.5, debye_temperature=300),
    ]
    lattice = Lattice(length_a=4.594, length_b=4.594, length_c=2.959, atom_sites=atom_sites)
    # Then
    cylinders = bond_cylinders(atom_spheres(lattice))
    # Expect
    # Ti-Ti atoms are 2.959 Å apart, which is within their covalent cutoff, but outside the first coordination
    # shell. Only the Ti-O bonds of about 2 Å remain.
    assert len(cylinders) > 0
    for cylinder in cylinders:
        assert 2 * math.dist(cylinder.start, cylinder.end) < 2.1


def test_bond_cylinders_without_atoms():
    # When Then Expect
    assert bond_cylinders(atom_spheres(Lattice.cubic(length_a=3.0))) == []


def test_lattice_vector_arrows():
    # When
    lattice = Lattice(length_a=1.0, length_b=2.0, length_c=3.0, alpha=80.0, beta=85.0, gamma=95.0)
    # Then
    arrows = lattice_vector_arrows(lattice)
    # Expect
    assert [arrow.label for arrow in arrows] == ['a', 'b', 'c']
    overhang = LATTICE_VECTOR_OVERHANG * 3.0
    for arrow, vector in zip(arrows, lattice.lattice_vectors):
        assert arrow.start == [0.0, 0.0, 0.0]
        assert np.allclose(np.cross(arrow.end, vector), 0.0)  # Along the lattice vector
        assert np.dot(arrow.end, vector) > 0  # In the same direction
        assert math.isclose(np.linalg.norm(arrow.end), np.linalg.norm(vector) + overhang)


def test_build_scene_can_be_converted_to_json():
    # When
    lattice = crystals.zincblende(length_a=5.43, atom1='Zn', atom2='S', debye_temperature1=300, debye_temperature2=300)
    # Then
    scene = build_scene(lattice)
    # Expect
    assert set(scene) == {'atoms', 'bonds', 'cell_edges', 'cell_edge_color', 'lattice_vectors'}
    json.dumps(scene)  # Raises a TypeError if the scene cannot be converted to JSON
