# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from .atom_site import AtomSite
from .lattice import Lattice

Numeric = int | float


def body_centered_cubic(
        length_a: Numeric,
        atomic_species: str,
        debye_temperature: Numeric | None = None,
        unique_name: str | None = None,
        display_name: str | None = None):
    """
    Create a body-centered cubic (BCC) lattice with two atom sites.
    """
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atomic_species=atomic_species,
                fract_x=0.0,
                fract_y=0.0,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atomic_species=atomic_species,
                fract_x=0.5,
                fract_y=0.5,
                fract_z=0.5,
                debye_temperature=debye_temperature,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice
