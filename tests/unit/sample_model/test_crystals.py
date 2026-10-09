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
    return [(site.fract_a.value, site.fract_b.value, site.fract_c.value) for site in lattice.atom_sites]


class TestBodyCenteredCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = body_centered_cubic(length_a=2.87, atom=Atoms.Fe)
        # Then Expect
        assert lattice.length_a.value == 2.87
        assert lattice.length_b.value == 2.87
        assert lattice.length_c.value == 2.87
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0
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

    def test_invalid_length_value(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length_a must be positive and non-zero.'):
            body_centered_cubic(length_a=0.0, atom=Atoms.Fe)

    def test_invalid_length_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Lattice length_a must be a numeric value'):
            body_centered_cubic(length_a='2.87', atom=Atoms.Fe)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string.'):
            body_centered_cubic(length_a=2.87, atom=255)


class TestFaceCenteredCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu)
        # Then Expect
        assert lattice.length_a.value == 3.6
        assert lattice.length_b.value == 3.6
        assert lattice.length_c.value == 3.6
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0
        assert len(lattice.atom_sites) == 4
        assert all(site.atom is Atoms.Cu for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ]

    def test_default_debye_temperature(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu)
        # Then Expect
        assert all(site.debye_temperature.value == 300.0 for site in lattice.atom_sites)

    def test_explicit_debye_temperature(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu, debye_temperature=470.0)
        # Then Expect
        assert all(site.debye_temperature.value == 470.0 for site in lattice.atom_sites)

    def test_default_naming(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu)
        # Then Expect
        assert lattice.unique_name.startswith('Cu FCC Lattice')

    def test_explicit_naming(self):
        # When
        lattice = face_centered_cubic(length_a=3.6, atom=Atoms.Cu, unique_name='my_fcc')
        # Then Expect
        assert lattice.unique_name == 'my_fcc'

    def test_invalid_length_value(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length_a must be positive and non-zero.'):
            face_centered_cubic(length_a=-1.0, atom=Atoms.Cu)

    def test_invalid_length_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Lattice length_a must be a numeric value'):
            face_centered_cubic(length_a='3.6', atom=Atoms.Cu)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string.'):
            face_centered_cubic(length_a=3.6, atom=127)


class TestDiamondCubic:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si)
        # Then Expect
        assert lattice.length_a.value == 5.43
        assert lattice.length_b.value == 5.43
        assert lattice.length_c.value == 5.43
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0
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

    def test_default_debye_temperature(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si)
        # Then Expect
        assert all(site.debye_temperature.value == 300.0 for site in lattice.atom_sites)

    def test_explicit_debye_temperature(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si, debye_temperature=470.0)
        # Then Expect
        assert all(site.debye_temperature.value == 470.0 for site in lattice.atom_sites)

    def test_default_naming(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si)
        # Then Expect
        assert lattice.unique_name.startswith('Si Diamond Cubic Lattice')

    def test_explicit_naming(self):
        # When
        lattice = diamond_cubic(length_a=5.43, atom=Atoms.Si, unique_name='my_diamond_cubic')
        # Then Expect
        assert lattice.unique_name == 'my_diamond_cubic'

    def test_invalid_length_value(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length_a must be positive and non-zero.'):
            diamond_cubic(length_a=0.0, atom=Atoms.Si)

    def test_invalid_length_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Lattice length_a must be a numeric value'):
            diamond_cubic(length_a='5.43', atom=Atoms.Si)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string.'):
            diamond_cubic(length_a=5.43, atom=64)


class TestZincblende:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As)
        # Then Expect
        assert lattice.length_a.value == 5.65
        assert lattice.length_b.value == 5.65
        assert lattice.length_c.value == 5.65
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0
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

    def test_default_debye_temperature(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As)
        # Then Expect
        assert all(site.debye_temperature.value == 300.0 for site in lattice.atom_sites)

    @pytest.mark.parametrize(
        'debye_temperature1, debye_temperature2',
        [(240.0, None), (None, 300.0), (240.0, 280.0)],
        ids=['first_debye_temperature', 'second_debye_temperature', 'both_debye_temperatures'],
    )
    def test_explicit_debye_temperatures_per_sublattice(self, debye_temperature1, debye_temperature2):
        # When
        lattice = zincblende(
            length_a=5.65,
            atom1=Atoms.Ga,
            atom2=Atoms.As,
            debye_temperature1=debye_temperature1,
            debye_temperature2=debye_temperature2,
        )
        # Then
        expected_values = []
        for dt in [debye_temperature1, debye_temperature2] * 4:
            expected_values.append(dt if dt is not None else 300.0)
        # Expect
        assert [site.debye_temperature.value for site in lattice.atom_sites] == expected_values

    def test_default_naming(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As)
        # Then Expect
        assert lattice.unique_name.startswith('GaAs Zincblende Lattice')

    def test_explicit_naming(self):
        # When
        lattice = zincblende(length_a=5.65, atom1=Atoms.Ga, atom2=Atoms.As, unique_name='my_zincblende')
        # Then Expect
        assert lattice.unique_name == 'my_zincblende'

    def test_invalid_length_value(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length_a must be positive and non-zero.'):
            zincblende(length_a=0.0, atom1=Atoms.Ga, atom2=Atoms.As)

    def test_invalid_length_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match='Lattice length_a must be a numeric value'):
            zincblende(length_a='5.65', atom1=Atoms.Ga, atom2=Atoms.As)

    @pytest.mark.parametrize(
        'atom1, atom2',
        [(255, Atoms.As), (Atoms.Ga, 127)],
        ids=['invalid_atom1', 'invalid_atom2'],
    )
    def test_invalid_atom(self, atom1, atom2):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string.'):
            zincblende(length_a=5.65, atom1=atom1, atom2=atom2)


class TestHexagonalClosePacked:
    def test_valid_lattice_and_sites(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=Atoms.Mg)
        # Then Expect
        assert lattice.length_a.value == 2.95
        assert lattice.length_b.value == 2.95
        assert lattice.length_c.value == 4.68
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 120.0
        assert len(lattice.atom_sites) == 2
        assert all(site.atom is Atoms.Mg for site in lattice.atom_sites)
        assert _fract_coords(lattice) == [(0.0, 0.0, 0.0), (pytest.approx(2 / 3), pytest.approx(1 / 3), 0.5)]

    def test_default_debye_temperature(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.87, length_c=4.68, atom=Atoms.Fe)
        # Then Expect
        assert all(site.debye_temperature.value == 300.0 for site in lattice.atom_sites)

    def test_explicit_debye_temperature(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.87, length_c=4.68, atom=Atoms.Fe, debye_temperature=470.0)
        # Then Expect
        assert all(site.debye_temperature.value == 470.0 for site in lattice.atom_sites)

    def test_default_naming(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=Atoms.Mg)
        # Then Expect
        assert lattice.unique_name.startswith('Mg HCP Lattice')

    def test_explicit_naming(self):
        # When
        lattice = hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=Atoms.Mg, unique_name='my_hcp')
        # Then Expect
        assert lattice.unique_name == 'my_hcp'

    def test_invalid_length_value(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length_a must be positive and non-zero.'):
            hexagonal_close_packed(length_a=0.0, length_c=4.68, atom=Atoms.Mg)

    @pytest.mark.parametrize('axis', ['a', 'c'])
    def test_invalid_length_type(self, axis):
        # When
        kwargs = dict(length_a=2.95, length_c=4.68, atom=Atoms.Mg)
        kwargs[f'length_{axis}'] = '1.0'
        # Then Expect
        with pytest.raises(TypeError, match=f'Lattice length_{axis} must be a numeric value'):
            hexagonal_close_packed(**kwargs)

    def test_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string.'):
            hexagonal_close_packed(length_a=2.95, length_c=4.68, atom=13)
