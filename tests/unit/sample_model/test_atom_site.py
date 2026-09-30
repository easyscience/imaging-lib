# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging

import numpy as np
import pytest
from easyscience import global_object

from easyimaging.sample_model import Atoms
from easyimaging.sample_model import AtomSite


class TestAtomSite:
    @pytest.fixture
    def atom_site(self):
        return AtomSite(
            atom=Atoms.Fe,
            fract_a=0.1,
            fract_b=0.2,
            fract_c=0.3,
            debye_temperature=250.0,
            unique_name=global_object.generate_unique_name('test_atom_site'),
            display_name='Test Atom Site',
        )

    def test_init_valid(self, atom_site):
        # When Then
        site = atom_site
        # Expect
        assert site.atom is Atoms.Fe
        assert site.fract_a.value == 0.1
        assert site.fract_b.value == 0.2
        assert site.fract_c.value == 0.3
        assert site.debye_temperature.value == 250.0
        assert site.unique_name.startswith('test_atom_site')
        assert site.display_name == 'Test Atom Site'

    def test_init_parameter_defaults(self, atom_site):
        # When
        site = atom_site
        # Then Expect
        assert site.fract_a.min == 0.0
        assert site.fract_a.max == 1.0
        assert site.fract_a.unit == 'dimensionless'
        assert site.fract_b.min == 0.0
        assert site.fract_b.max == 1.0
        assert site.fract_b.unit == 'dimensionless'
        assert site.fract_c.min == 0.0
        assert site.fract_c.max == 1.0
        assert site.fract_c.unit == 'dimensionless'
        assert site.debye_temperature.min == 0.0
        assert site.debye_temperature.max == np.inf
        assert site.debye_temperature.unit == 'K'

    def test_init_default_unique_name(self):
        # When
        site = AtomSite(atom=Atoms.Fe, fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=1.0)
        # Then
        unique_name = site.unique_name
        # Expect
        assert site.unique_name.startswith('Fe AtomSite')
        assert site.display_name == site.unique_name
        assert site.debye_temperature.unique_name.startswith(unique_name + '_debye_temperature')
        assert site.fract_a.unique_name.startswith(unique_name + '_fract_a')
        assert site.fract_b.unique_name.startswith(unique_name + '_fract_b')
        assert site.fract_c.unique_name.startswith(unique_name + '_fract_c')

    def test_init_missing_debye_temperature_defaults_and_warns(self, caplog):
        # When
        with caplog.at_level(logging.WARNING):
            site = AtomSite(atom=Atoms.Fe, fract_a=0.0, fract_b=0.0, fract_c=0.0)
        # Then Expect
        assert site.debye_temperature.value == 300.0
        assert any('Debye temperature not provided' in record.message for record in caplog.records)

    def test_init_atom_from_string(self):
        # When
        site = AtomSite(atom='Fe', fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=1.0)
        # Then Expect
        assert site.atom is Atoms.Fe
        assert site.unique_name.startswith('Fe AtomSite')

    @pytest.mark.parametrize('invalid_atom', ['not_an_atom', 'fe', ''], ids=['unknown', 'wrong_case', 'empty'])
    def test_init_invalid_atom_string(self, invalid_atom):
        # When Then Expect
        with pytest.raises(KeyError, match='"atom" must be a valid Atoms enum or a valid Atoms name string'):
            AtomSite(atom=invalid_atom, fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=1.0)

    @pytest.mark.parametrize('invalid_atom', [26, None, 1.5], ids=['int', 'none', 'float'])
    def test_init_invalid_atom_type(self, invalid_atom):
        # When Then Expect
        with pytest.raises(TypeError, match='"atom" must be a valid Atoms enum or a valid Atoms name string'):
            AtomSite(atom=invalid_atom, fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=1.0)

    @pytest.mark.parametrize('axis', ['a', 'b', 'c'], ids=['a', 'b', 'c'])
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
        kwargs = {'atom': Atoms.Fe, 'fract_a': 0.0, 'fract_b': 0.0, 'fract_c': 0.0, 'debye_temperature': 1.0}
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
            AtomSite(atom=Atoms.Fe, fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=invalid_value)

    def test_atom_setter_valid(self, atom_site):
        # When
        site = atom_site
        # Then
        site.atom = Atoms.Co
        # Expect
        assert site.atom is Atoms.Co

    def test_atom_setter_valid_string(self, atom_site):
        # When
        site = atom_site
        # Then
        site.atom = 'Co'
        # Expect
        assert site.atom is Atoms.Co

    @pytest.mark.parametrize('new_atom', [Atoms.Co, 'Co'], ids=['enum', 'string'])
    def test_atom_setter_regenerates_default_unique_name(self, new_atom):
        # When
        site = AtomSite(atom=Atoms.Fe, fract_a=0.0, fract_b=0.0, fract_c=0.0, debye_temperature=1.0)
        # Then
        site.atom = new_atom
        # Expect
        assert site.unique_name.startswith('Co AtomSite')

    def test_atom_setter_keeps_explicit_unique_name(self, atom_site):
        # When
        site = atom_site
        # Then
        site.atom = Atoms.Co
        # Expect
        assert site.unique_name.startswith('test_atom_site')

    @pytest.mark.parametrize(
        'invalid_atom, error',
        [
            ('not_an_atom', KeyError),
            (26, TypeError),
            (None, TypeError),
        ],
        ids=['invalid_string', 'int', 'none'],
    )
    def test_atom_setter_invalid_leaves_atom_unchanged(self, atom_site, invalid_atom, error):
        # When
        site = atom_site
        original_unique_name = site.unique_name
        # Then Expect
        with pytest.raises(error, match='"atom" must be a valid Atoms enum or a valid Atoms name string'):
            site.atom = invalid_atom
        assert site.atom is Atoms.Fe
        assert site.unique_name == original_unique_name

    @pytest.mark.parametrize('attribute', ['fract_a', 'fract_b', 'fract_c'])
    def test_fract_setter_valid(self, atom_site, attribute):
        # When
        site = atom_site
        # Then
        setattr(site, attribute, 0.75)
        # Expect
        assert getattr(site, attribute).value == 0.75

    @pytest.mark.parametrize('attribute', ['fract_a', 'fract_b', 'fract_c'])
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
