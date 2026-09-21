# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyimaging.sample_model.atoms import Atoms
from easyimaging.sample_model.crystals import body_centered_cubic
from easyimaging.sample_model.crystals import diamond_cubic
from easyimaging.sample_model.crystals import face_centered_cubic
from easyimaging.sample_model.crystals import hexagonal_close_packed
from easyimaging.sample_model.crystals import zincblende


def _fract_coords(lattice):
    return [(site.fract_x.value, site.fract_y.value, site.fract_z.value) for site in lattice.atom_sites]


class TestBodyCenteredCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe)
        # Then Expect
        assert lattice.length_a.value == 2.87
        assert lattice.length_b.value == 2.87
        assert lattice.length_c.value == 2.87
        assert len(lattice.atom_sites) == 2
        assert all(site.atom is Atoms.Fe for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)]

    def test_default_debye_temperature(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe)
        # Then Expect
        assert all(site.debye_temperature.value == 300.0 for site in lattice.atom_sites)

    def test_explicit_debye_temperature(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe, debye_temperature=470.0)
        # Then Expect
        assert all(site.debye_temperature.value == 470.0 for site in lattice.atom_sites)

    def test_default_naming(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe)
        # Then Expect
        assert lattice.unique_name.startswith('Fe BCC Lattice')

    def test_explicit_naming(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe, unique_name='my_bcc', display_name='My BCC')
        # Then Expect
        assert lattice.unique_name == 'my_bcc'
        assert lattice.display_name == 'My BCC'

    def test_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive.'):
            body_centered_cubic(length_a=0.0, atom=Atoms.Fe)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            body_centered_cubic(length_a=2.87, atom='Fe')


class TestFaceCenteredCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu)
        # Then Expect
        assert lattice.length_a.value == 3.6
        assert len(lattice.atom_sites) == 4
        assert all(site.atom is Atoms.Cu for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ]

    def test_default_naming(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu)
        # Then Expect
        assert lattice.unique_name.startswith('Cu FCC Lattice')

    def test_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive.'):
            face_centered_cubic(length_a=-1.0, atom=Atoms.Cu)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            face_centered_cubic(length_a=3.6, atom='Cu')


class TestDiamondCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si)
        # Then Expect
        assert lattice.length_a.value == 5.43
        assert len(lattice.atom_sites) == 8
        assert all(site.atom is Atoms.Si for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.25),
            (0.5, 0.5, 0.0),
            (0.75, 0.75, 0.25),
            (0.5, 0.0, 0.5),
            (0.75, 0.25, 0.75),
            (0.0, 0.5, 0.5),
            (0.25, 0.75, 0.75),
        ]

    def test_default_naming(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si)
        # Then Expect
        assert lattice.unique_name.startswith('Si Diamond Cubic Lattice')

    def test_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive.'):
            diamond_cubic(length_a=0.0, atom=Atoms.Si)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            diamond_cubic(length_a=5.43, atom='Si')


class TestZincblende:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As)
        # Then Expect
        assert lattice.length_a.value == 5.65
        assert len(lattice.atom_sites) == 8
        assert [site.atom for site in lattice.atom_sites] == [
            Atoms.Ga,
            Atoms.As,
            Atoms.Ga,
            Atoms.As,
            Atoms.Ga,
            Atoms.As,
            Atoms.Ga,
            Atoms.As,
        ]
        assert _fract_coords(lattice) == [
            (0.0, 0.0, 0.0),
            (0.25, 0.25, 0.25),
            (0.5, 0.5, 0.0),
            (0.75, 0.75, 0.25),
            (0.5, 0.0, 0.5),
            (0.75, 0.25, 0.75),
            (0.0, 0.5, 0.5),
            (0.25, 0.75, 0.75),
        ]

    def test_debye_temperatures_per_sublattice(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As, debye_temperature1=240.0, debye_temperature2=280.0)
        # Then Expect
        assert [site.debye_temperature.value for site in lattice.atom_sites] == [
            240.0,
            280.0,
            240.0,
            280.0,
            240.0,
            280.0,
            240.0,
            280.0,
        ]

    def test_default_naming(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As)
        # Then Expect
        assert lattice.unique_name.startswith('GaAs Zincblende Lattice')

    def test_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive.'):
            zincblende(length_a=0.0, atom1=Atoms.Ga, atom2=Atoms.As)

    @pytest.mark.parametrize(
        'atom1, atom2',
        [('Ga', Atoms.As), (Atoms.Ga, 'As')],
        ids=['invalid_atom1', 'invalid_atom2'],
    )
    def test_invalid_atom(self, atom1, atom2):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            zincblende(length_a=5.65, atom1=atom1, atom2=atom2)


class TestHexagonalClosePacked:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=Atoms.Mg)
        # Then Expect
        assert lattice.length_a.value == 2.95
        assert lattice.length_b.value == 2.95
        assert lattice.length_c.value == 4.68
        assert lattice.gamma.value == 120.0
        assert len(lattice.atom_sites) == 2
        assert all(site.atom is Atoms.Mg for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [(0.0, 0.0, 0.0), (pytest.approx(2 / 3), pytest.approx(1 / 3), 0.5)]

    def test_default_naming(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=Atoms.Mg)
        # Then Expect
        assert lattice.unique_name.startswith('Mg HCP Lattice')

    def test_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive.'):
            hexagonal_close_packed(length_a=0.0, length_c=4.68, atom=Atoms.Mg)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            hexagonal_close_packed(length_a=2.95, length_c=4.68, atom='Mg')
