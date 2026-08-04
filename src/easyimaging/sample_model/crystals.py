# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from .atom_site import AtomSite
from .atoms import Atoms
from .lattice import Lattice

Numeric = int | float


def body_centered_cubic(
        length_a: Numeric,
        atom: Atoms,
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
                atom=atom,
                fract_x=0.0,
                fract_y=0.0,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
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


def face_centered_cubic(
        length_a: Numeric,
        atom: Atoms,
        debye_temperature: Numeric | None = None,
        unique_name: str | None = None,
        display_name: str | None = None):
    """
    Create a face-centered cubic (FCC) lattice with four atom sites.
    """
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_x=0.0,
                fract_y=0.0,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.5,
                fract_y=0.5,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.5,
                fract_y=0.0,
                fract_z=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.0,
                fract_y=0.5,
                fract_z=0.5,
                debye_temperature=debye_temperature,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def diamond_cubic(
        length_a: Numeric,
        atom: Atoms,
        debye_temperature: Numeric | None = None,
        unique_name: str | None = None,
        display_name: str | None = None):
    """
    Create a diamond cubic lattice with eight atom sites.
    """
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_x=0.0,
                fract_y=0.0,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.25,
                fract_y=0.25,
                fract_z=0.25,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.5,
                fract_y=0.5,
                fract_z=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.75,
                fract_y=0.75,
                fract_z=0.25,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.5,
                fract_y=0.0,
                fract_z=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.75,
                fract_y=0.25,
                fract_z=0.75,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.0,
                fract_y=0.5,
                fract_z=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_x=0.25,
                fract_y=0.75,
                fract_z=0.75,
                debye_temperature=debye_temperature,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def zincblende(
        length_a: Numeric,
        atom1: Atoms,
        atom2: Atoms,
        debye_temperature1: Numeric | None = None,
        debye_temperature2: Numeric | None = None,
        unique_name: str | None = None,
        display_name: str | None = None):
    """
    Create a zincblende lattice with eight atom sites.
    """
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom1,
                fract_x=0.0,
                fract_y=0.0,
                fract_z=0.0,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_x=0.25,
                fract_y=0.25,
                fract_z=0.25,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_x=0.5,
                fract_y=0.5,
                fract_z=0.0,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_x=0.75,
                fract_y=0.75,
                fract_z=0.25,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_x=0.5,
                fract_y=0.0,
                fract_z=0.5,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_x=0.75,
                fract_y=0.25,
                fract_z=0.75,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_x=0.0,
                fract_y=0.5,
                fract_z=0.5,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_x=0.25,
                fract_y=0.75,
                fract_z=0.75,
                debye_temperature=debye_temperature2,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name
    )
    return lattice

