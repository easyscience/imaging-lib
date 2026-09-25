# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyimaging.sample_model.atom_site import AtomSite
from easyimaging.sample_model.atoms import Atoms
from easyimaging.sample_model.lattice import INFINITESIMAL
from easyimaging.sample_model.lattice import Lattice


class TestLattice:
    @pytest.fixture
    def atom_sites(self):
        return [
            AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=1.0),
            AtomSite(atom=Atoms.Fe, fract_x=0.5, fract_y=0.5, fract_z=0.5, debye_temperature=1.0),
        ]

    @pytest.fixture
    def lattice(self, atom_sites):
        return Lattice(
            length_a=1.0,
            length_b=2.0,
            length_c=3.0,
            alpha=80.0,
            beta=85.0,
            gamma=95.0,
            atom_sites=atom_sites,
            unique_name='test_lattice',
            display_name='Test Lattice',
        )

    def test_init_valid(self, lattice, atom_sites):
        # When Then Expect
        assert lattice.length_a.value == 1.0
        assert lattice.length_b.value == 2.0
        assert lattice.length_c.value == 3.0
        assert lattice.alpha.value == 80.0
        assert lattice.beta.value == 85.0
        assert lattice.gamma.value == 95.0
        assert lattice.unique_name == 'test_lattice'
        assert lattice.display_name == 'Test Lattice'
        assert len(lattice.atom_sites) == 2
        assert lattice.atom_sites[0] is atom_sites[0]
        assert lattice.atom_sites[1] is atom_sites[1]

    def test_init_parameter_defaults(self):
        # When
        lattice = Lattice(length_a=1.0, length_b=2.0, length_c=3.0)
        # Then Expect
        assert lattice.length_a.min == 1e-3
        assert lattice.length_b.min == 1e-3
        assert lattice.length_c.min == 1e-3
        assert lattice.length_a.unit == 'Å'
        assert lattice.length_b.unit == 'Å'
        assert lattice.length_c.unit == 'Å'
        assert lattice.alpha.min == INFINITESIMAL
        assert lattice.alpha.max == 180 - INFINITESIMAL
        assert lattice.alpha.unit == 'deg'
        assert lattice.beta.min == INFINITESIMAL
        assert lattice.beta.max == 180 - INFINITESIMAL
        assert lattice.beta.unit == 'deg'
        assert lattice.gamma.min == INFINITESIMAL
        assert lattice.gamma.max == 180 - INFINITESIMAL
        assert lattice.gamma.unit == 'deg'

    def test_init_default_angles(self):
        # When Then
        lattice = Lattice(length_a=1.0, length_b=2.0, length_c=3.0)
        # Expect
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0

    def test_init_default_unique_name(self, atom_sites):
        # When
        lattice = Lattice(length_a=1.0, length_b=1.0, length_c=1.0, atom_sites=atom_sites)
        # Then
        unique_name = lattice.unique_name
        # Expect
        assert lattice.unique_name.startswith('Lattice')
        assert lattice.display_name == lattice.unique_name
        assert lattice.alpha.unique_name.startswith(unique_name + '_angle_alpha')
        assert lattice.beta.unique_name.startswith(unique_name + '_angle_beta')
        assert lattice.gamma.unique_name.startswith(unique_name + '_angle_gamma')
        assert lattice.length_a.unique_name.startswith(unique_name + '_length_a')
        assert lattice.length_b.unique_name.startswith(unique_name + '_length_b')
        assert lattice.length_c.unique_name.startswith(unique_name + '_length_c')

    def test_init_no_atom_sites_is_empty(self):
        # When Then
        lattice = Lattice(length_a=1.0, length_b=1.0, length_c=1.0)
        # Expect
        assert len(lattice.atom_sites) == 0

    @pytest.mark.parametrize('length_a, length_b, length_c', [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, -1.0)])
    def test_init_invalid_lengths(self, length_a, length_b, length_c):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive and non-zero.'):
            Lattice(length_a=length_a, length_b=length_b, length_c=length_c)

    @pytest.mark.parametrize(
        'alpha, beta, gamma', [(0.0, 90.0, 90.0), (180.0, 90.0, 90.0), (90.0, -10.0, 90.0), (90.0, 90.0, 200.0)]
    )
    def test_init_invalid_angles(self, alpha, beta, gamma):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice angles alpha, beta, and gamma must be between 0 and 180 degrees.'):
            Lattice(length_a=1.0, length_b=1.0, length_c=1.0, alpha=alpha, beta=beta, gamma=gamma)

    @pytest.mark.parametrize('atom_sites', ['not_a_list', ['not_an_atom_site']], ids=['not_a_list', 'not_an_atom_site'])
    def test_init_invalid_atom_sites_type(self, atom_sites):
        # When Then Expect
        with pytest.raises(TypeError, match='atom_sites must be a list of AtomSite objects.'):
            Lattice(length_a=1.0, length_b=1.0, length_c=1.0, atom_sites=atom_sites)

    def test_cubic_valid(self):
        # When
        lattice = Lattice.cubic(length_a=3.0)
        # Then Expect
        assert lattice.length_a.value == 3.0
        assert lattice.length_b.value == 3.0
        assert lattice.length_c.value == 3.0
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 90.0
        assert lattice.unique_name.startswith('CubicLattice')

    def test_cubic_length_b_and_c_track_length_a(self):
        # When
        lattice = Lattice.cubic(length_a=3.0)
        # Then
        lattice.length_a = 5.0
        # Expect
        assert lattice.length_b.value == 5.0
        assert lattice.length_c.value == 5.0

    def test_cubic_with_atom_sites(self, atom_sites):
        # When
        lattice = Lattice.cubic(length_a=3.0, atom_sites=atom_sites)
        # Then Expect
        assert len(lattice.atom_sites) == 2

    def test_cubic_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive and non-zero.'):
            Lattice.cubic(length_a=0.0)

    def test_hexagonal_valid(self):
        # When
        lattice = Lattice.hexagonal(length_a=2.0, length_c=4.0)
        # Then Expect
        assert lattice.length_a.value == 2.0
        assert lattice.length_b.value == 2.0
        assert lattice.length_c.value == 4.0
        assert lattice.alpha.value == 90.0
        assert lattice.beta.value == 90.0
        assert lattice.gamma.value == 120.0
        assert lattice.unique_name.startswith('HexagonalLattice')

    def test_hexagonal_length_b_tracks_length_a_only(self):
        # When
        lattice = Lattice.hexagonal(length_a=2.0, length_c=4.0)
        # Then
        lattice.length_a = 6.0
        lattice.length_c = 5.0
        # Expect
        assert lattice.length_b.value == 6.0
        assert lattice.length_c.value == 5.0

    def test_hexagonal_invalid_length(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice lengths must be positive and non-zero.'):
            Lattice.hexagonal(length_a=1.0, length_c=0.0)

    @pytest.mark.parametrize('attribute', ['length_a', 'length_b', 'length_c'])
    def test_length_setter_valid(self, lattice, attribute):
        # When Then
        setattr(lattice, attribute, 9.0)
        # Expect
        assert getattr(lattice, attribute).value == 9.0

    @pytest.mark.parametrize('invalid_value', [0.0, -1.0])
    @pytest.mark.parametrize('attribute', ['length_a', 'length_b', 'length_c'])
    def test_length_setter_invalid(self, lattice, attribute, invalid_value):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice length must be positive and non-zero.'):
            setattr(lattice, attribute, invalid_value)

    @pytest.mark.parametrize('attribute', ['alpha', 'beta', 'gamma'])
    def test_angle_setter_valid(self, lattice, attribute):
        # When Then
        setattr(lattice, attribute, 100.0)
        # Expect
        assert getattr(lattice, attribute).value == 100.0

    @pytest.mark.parametrize('attribute', ['alpha', 'beta', 'gamma'])
    @pytest.mark.parametrize('invalid_value', [0.0, 180.0, 250.0])
    def test_angle_setter_invalid(self, lattice, attribute, invalid_value):
        # When Then Expect
        with pytest.raises(ValueError, match='Lattice angle must be between 0 and 180 degrees.'):
            setattr(lattice, attribute, invalid_value)
