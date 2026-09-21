# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from easyimaging.sample_model.atoms import Atom
from easyimaging.sample_model.atoms import Atoms


class TestAtom:
    def test_init_stores_fields(self):
        # When
        atom = Atom(element='Fe', isotope=56, enum_id=263)
        # Then Expect
        assert atom.element == 'Fe'
        assert atom.isotope == 56
        assert atom.enum_id == 263

    def test_init_natural_abundance_isotope_is_none(self):
        # When
        atom = Atom(element='Fe', isotope=None, enum_id=261)
        # Then Expect
        assert atom.isotope is None


class TestAtoms:
    @pytest.mark.parametrize(
        'member, element, isotope, enum_id',
        [
            (Atoms.H, 'H', None, 11),
            (Atoms.H2, 'H', 2, 13),
            (Atoms.Fe, 'Fe', None, 261),
            (Atoms.Fe56, 'Fe', 56, 263),
            (Atoms.U238, 'U', 238, 875),
        ],
        ids=['natural_hydrogen', 'deuterium', 'natural_iron', 'iron_56', 'uranium_238'],
    )
    def test_member_attributes(self, member, element, isotope, enum_id):
        # When Then Expect
        assert member.element == element
        assert member.isotope == isotope
        assert member.enum_id == enum_id

    def test_membership_valid_member(self):
        # When Then Expect
        assert Atoms.Fe in Atoms

    @pytest.mark.parametrize('value', ['Fe', None, 42], ids=['string', 'none', 'integer'])
    def test_membership_invalid_value(self, value):
        # When Then Expect
        assert value not in Atoms

    def test_lookup_by_member_name(self):
        # When
        member = Atoms['Fe56']
        # Then Expect
        assert member is Atoms.Fe56

    def test_enum_ids_are_unique_across_all_members(self):
        # When
        enum_ids = [member.enum_id for member in Atoms]
        # Then Expect
        assert len(enum_ids) == len(set(enum_ids))
