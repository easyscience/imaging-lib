# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import pytest

from easyimaging.sample_model import Atoms
from easyimaging.sample_model import AtomSite


class TestAtomSite:
    @pytest.fixture
    def atom_site(self):
        return AtomSite(
            atom=Atoms.Fe,
            fract_x=0.1,
            fract_y=0.2,
            fract_z=0.3,
            debye_temperature=250.0,
            unique_name='test_atom_site',
            display_name='Test Atom Site',
        )

    def test_init_valid(self, atom_site):
        # When Then
        site = atom_site
        # Expect
        assert site.atom is Atoms.Fe
        assert site.fract_x.value == 0.1
        assert site.fract_y.value == 0.2
        assert site.fract_z.value == 0.3
        assert site.debye_temperature.value == 250.0
        assert site.unique_name == 'test_atom_site'
        assert site.display_name == 'Test Atom Site'

    def test_init_parameter_defaults(self, atom_site):
        # When
        site = atom_site
        # Then Expect
        assert site.fract_x.min == 0.0
        assert site.fract_x.max == 1.0
        assert site.fract_x.unit == 'dimensionless'
        assert site.fract_y.min == 0.0
        assert site.fract_y.max == 1.0
        assert site.fract_y.unit == 'dimensionless'
        assert site.fract_z.min == 0.0
        assert site.fract_z.max == 1.0
        assert site.fract_z.unit == 'dimensionless'
        assert site.debye_temperature.min == 0.0
        assert site.debye_temperature.max == np.inf
        assert site.debye_temperature.unit == 'K'

    def test_init_default_unique_name(self):
        # When
        site = AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=1.0)
        # Then
        unique_name = site.unique_name
        # Expect
        assert site.unique_name.startswith('Fe AtomSite')
        assert site.display_name == site.unique_name
        assert site.debye_temperature.unique_name.startswith(unique_name + '_debye_temperature')
        assert site.fract_x.unique_name.startswith(unique_name + '_fract_x')
        assert site.fract_y.unique_name.startswith(unique_name + '_fract_y')
        assert site.fract_z.unique_name.startswith(unique_name + '_fract_z')

    def test_init_missing_debye_temperature_defaults_and_warns(self, caplog):
        # When
        with caplog.at_level(logging.WARNING):
            site = AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0)
        # Then Expect
        assert site.debye_temperature.value == 300.0
        assert any('Debye temperature not provided' in record.message for record in caplog.records)

    def test_init_invalid_atom(self):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            AtomSite(atom='Fe', fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=1.0)

    @pytest.mark.parametrize('axis', ['x', 'y', 'z'], ids=['x', 'y', 'z'])
    @pytest.mark.parametrize(
        'invalid_value, error, message',
        [
            (1.5, ValueError, r'must be between 0\.0 and 1\.0'),
            (-0.1, ValueError, r'must be between 0\.0 and 1\.0'),
            ('0.5', TypeError, 'must be a numeric value'),
        ],
        ids=['above_range', 'below_range', 'wrong_type'],
    )
    def test_init_invalid_fract_value(self, axis, invalid_value, error, message):
        # When
        kwargs = dict(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=1.0)
        kwargs[f'fract_{axis}'] = invalid_value
        # Then Expect
        with pytest.raises(error, match=message):
            AtomSite(**kwargs)

    @pytest.mark.parametrize(
        'invalid_value, error, message',
        [
            (-1.0, ValueError, 'must be non-negative'),
            ('cold', TypeError, 'must be a numeric value'),
        ],
        ids=['negative', 'wrong_type'],
    )
    def test_init_invalid_debye_temperature(self, invalid_value, error, message):
        # When Then Expect
        with pytest.raises(error, match=message):
            AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=invalid_value)

    def test_atom_setter_valid(self, atom_site):
        # When
        site = atom_site
        # Then
        site.atom = Atoms.Co
        # Expect
        assert site.atom is Atoms.Co

    def test_atom_setter_regenerates_default_unique_name(self):
        # When
        site = AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=1.0)
        # Then
        site.atom = Atoms.Co
        # Expect
        assert site.unique_name.startswith('Co AtomSite')

    def test_atom_setter_keeps_explicit_unique_name(self, atom_site):
        # When
        site = atom_site
        # Then
        site.atom = Atoms.Co
        # Expect
        assert site.unique_name == 'test_atom_site'

    def test_atom_setter_invalid_leaves_atom_unchanged(self, atom_site):
        # When
        site = atom_site
        # Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum'):
            site.atom = 'not_an_atom'
        assert site.atom is Atoms.Fe

    @pytest.mark.parametrize('attribute', ['fract_x', 'fract_y', 'fract_z'])
    def test_fract_setter_valid(self, atom_site, attribute):
        # When
        site = atom_site
        # Then
        setattr(site, attribute, 0.75)
        # Expect
        assert getattr(site, attribute).value == 0.75

    @pytest.mark.parametrize('attribute', ['fract_x', 'fract_y', 'fract_z'])
    def test_fract_setter_invalid_leaves_value_unchanged(self, atom_site, attribute):
        # When
        site = atom_site
        original_value = getattr(site, attribute).value
        # Then Expect
        with pytest.raises(ValueError, match=r'must be between 0\.0 and 1\.0'):
            setattr(site, attribute, 2.0)
        assert getattr(site, attribute).value == original_value

    def test_debye_temperature_setter_valid(self, atom_site):
        # When
        site = atom_site
        # Then
        site.debye_temperature = 400.0
        # Expect
        assert site.debye_temperature.value == 400.0

    def test_debye_temperature_setter_invalid_leaves_value_unchanged(self, atom_site):
        # When
        site = atom_site
        # Then Expect
        with pytest.raises(ValueError, match='must be non-negative'):
            site.debye_temperature = -10.0
        assert site.debye_temperature.value == 250.0

    def test_repr(self, atom_site):
        # When Then Expect
        assert repr(atom_site) == "AtomSite(atomic_species='Fe', fract_x=0.1, fract_y=0.2, fract_z=0.3)"
