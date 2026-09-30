# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object

from .atom_site import AtomSite
from .atoms import Atoms
from .lattice import Lattice

Numeric = int | float


def body_centered_cubic(
    length_a: Numeric,
    atom: Atoms,
    debye_temperature: Numeric | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
):
    """
    Create a body-centered cubic (BCC) lattice with two atom sites.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in angstrom.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'BCC Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the BCC lattice with the specified parameters.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If ``atom`` is neither an [`Atoms`][.atoms.Atoms] enum member nor a string.
    KeyError
        If ``atom`` is a string that does not correspond to a valid [`Atoms`][.atoms.Atoms] member.
    """
    if unique_name is None:
        name = getattr(atom, '_name_', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} BCC Lattice')
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.0,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.5,
                fract_b=0.5,
                fract_c=0.5,
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
    display_name: str | None = None,
):
    """
    Create a face-centered cubic (FCC) lattice with four atom sites.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in angstrom.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'FCC Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the FCC lattice with the specified parameters.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If ``atom`` is neither an [`Atoms`][.atoms.Atoms] enum member nor a string.
    KeyError
        If ``atom`` is a string that does not correspond to a valid [`Atoms`][.atoms.Atoms] member.
    """
    if unique_name is None:
        name = getattr(atom, '_name_', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} FCC Lattice')
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.0,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.5,
                fract_b=0.5,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.5,
                fract_b=0.0,
                fract_c=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.5,
                fract_c=0.5,
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
    display_name: str | None = None,
):
    """
    Create a diamond cubic lattice with eight atom sites.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in angstrom.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'Diamond Cubic Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the diamond cubic lattice with the specified parameters.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If ``atom`` is neither an [`Atoms`][.atoms.Atoms] enum member nor a string.
    KeyError
        If ``atom`` is a string that does not correspond to a valid [`Atoms`][.atoms.Atoms] member.
    """
    if unique_name is None:
        name = getattr(atom, '_name_', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} Diamond Cubic Lattice')
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.0,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.25,
                fract_b=0.25,
                fract_c=0.25,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.5,
                fract_b=0.5,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.75,
                fract_b=0.75,
                fract_c=0.25,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.5,
                fract_b=0.0,
                fract_c=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.75,
                fract_b=0.25,
                fract_c=0.75,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.5,
                fract_c=0.5,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=0.25,
                fract_b=0.75,
                fract_c=0.75,
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
    display_name: str | None = None,
):
    """
    Create a zincblende lattice with eight atom sites.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in angstrom.
    atom1 : Atoms | str
        The atomic species of the first FCC sub-lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name as
        a string.
    atom2 : Atoms | str
        The atomic species of the second FCC sub-lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name
        as a string.
    debye_temperature1 : float | int | None
        The Debye temperature of the first sub-lattice atomic species in Kelvin. If None, a default value of 300 K is used.
    debye_temperature2 : float | int | None
        The Debye temperature of the second sub-lattice atomic species in Kelvin. If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'Zincblende Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the zincblende lattice with the specified parameters.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If ``atom1`` or ``atom2`` is neither an [`Atoms`][.atoms.Atoms] enum member nor a string.
    KeyError
        If ``atom1`` or ``atom2`` is a string that does not correspond to a valid [`Atoms`][.atoms.Atoms] member.
    """
    if unique_name is None:
        name1 = getattr(atom1, '_name_', 'Unknown')
        name2 = getattr(atom2, '_name_', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name1}{name2} Zincblende Lattice')
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[
            AtomSite(
                atom=atom1,
                fract_a=0.0,
                fract_b=0.0,
                fract_c=0.0,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_a=0.25,
                fract_b=0.25,
                fract_c=0.25,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_a=0.5,
                fract_b=0.5,
                fract_c=0.0,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_a=0.75,
                fract_b=0.75,
                fract_c=0.25,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_a=0.5,
                fract_b=0.0,
                fract_c=0.5,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_a=0.75,
                fract_b=0.25,
                fract_c=0.75,
                debye_temperature=debye_temperature2,
            ),
            AtomSite(
                atom=atom1,
                fract_a=0.0,
                fract_b=0.5,
                fract_c=0.5,
                debye_temperature=debye_temperature1,
            ),
            AtomSite(
                atom=atom2,
                fract_a=0.25,
                fract_b=0.75,
                fract_c=0.75,
                debye_temperature=debye_temperature2,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def hexagonal_close_packed(
    length_a: Numeric,
    length_c: Numeric,
    atom: Atoms,
    debye_temperature: Numeric | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
):
    """
    Create a hexagonal close-packed (HCP) lattice with four atom sites.

    Parameters
    ----------
    length_a : float | int
        The length of the a-axis lattice vector in angstrom.
    length_c : float | int
        The length of the c-axis lattice vector in angstrom.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][.atoms.Atoms] enum member or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'HCP Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the HCP lattice with the specified parameters.

    Raises
    ------
    ValueError
        If ``length_a`` or ``length_c`` is not positive.
    TypeError
        If ``atom`` is neither an [`Atoms`][.atoms.Atoms] enum member nor a string.
    KeyError
        If ``atom`` is a string that does not correspond to a valid [`Atoms`][.atoms.Atoms] member.
    """
    if unique_name is None:
        name = getattr(atom, '_name_', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} HCP Lattice')
    lattice = Lattice.hexagonal(
        length_a=length_a,
        length_c=length_c,
        atom_sites=[
            AtomSite(
                atom=atom,
                fract_a=0.0,
                fract_b=0.0,
                fract_c=0.0,
                debye_temperature=debye_temperature,
            ),
            AtomSite(
                atom=atom,
                fract_a=2 / 3,
                fract_b=1 / 3,
                fract_c=0.5,
                debye_temperature=debye_temperature,
            ),
        ],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice
