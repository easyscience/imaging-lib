# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""
This module contains convenience functions to easily create [`Lattice`][easyimaging.sample_model.Lattice] objects and occupy
them with [`Atoms`][easyimaging.sample_model.atoms.Atoms] to create common crystallographic lattices."""

import numbers

from easyscience import global_object

from .atom_site import AtomSite
from .atoms import Atoms
from .lattice import Lattice

_BCC = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
_FCC = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5))
_HCP = ((0.0, 0.0, 0.0), (2 / 3, 1 / 3, 0.5))


def body_centered_cubic(
    length_a: numbers.Real,
    atom: Atoms | str,
    debye_temperature: numbers.Real | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
) -> Lattice:
    """
    Create a body-centered cubic (BCC) lattice.

    This function creates a [`Lattice`][...Lattice] object with lattice constants $a=b=c$ and lattice angles
    $\\alpha=\\beta=\\gamma=90^\\circ$. The [`Lattice`][...Lattice] object is then populated with two atoms at the fractional
    coordinates: $(0,0,0)$ and $(\\frac{1}{2},\\frac{1}{2},\\frac{1}{2})$, corresponding to a BCC lattice.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in Å.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. <br> If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'BCC Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to the [`unique_name`][...Lattice.unique_name] if not
        provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the BCC lattice.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If `atom` is not an [`Atoms`][...atoms.Atoms] enum member or a string.
    KeyError
        If the provided atom string does not correspond to a valid [`Atoms`][...atoms.Atoms] enum member.
    """
    if unique_name is None:
        name = atom if isinstance(atom, str) else getattr(atom, 'name', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} BCC Lattice')
    if debye_temperature is None:
        global_object.log.warning(f"Debye temperature not provided for '{unique_name}'. Setting to default value of 300 K.")
        debye_temperature = 300.0
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[AtomSite(atom, *xyz, debye_temperature=debye_temperature) for xyz in _BCC],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def face_centered_cubic(
    length_a: numbers.Real,
    atom: Atoms | str,
    debye_temperature: numbers.Real | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
) -> Lattice:
    """
    Create a face-centered cubic (FCC) lattice.

    This function creates a [`Lattice`][...Lattice] object with lattice constants $a=b=c$ and lattice angles
    $\\alpha=\\beta=\\gamma=90^\\circ$. The [`Lattice`][...Lattice] object is then populated with four atoms at the fractional
    coordinates: $(0,0,0)$, $(\\frac{1}{2},\\frac{1}{2},0)$, $(\\frac{1}{2},0,\\frac{1}{2})$, and
    $(0,\\frac{1}{2},\\frac{1}{2})$, corresponding to an FCC lattice.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in Å.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. <br> If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'FCC Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to the [`unique_name`][...Lattice.unique_name] if not
        provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the FCC lattice.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If `atom` is not an [`Atoms`][...atoms.Atoms] enum member or a string.
    KeyError
        If the provided atom string does not correspond to a valid [`Atoms`][...atoms.Atoms] enum member.
    """
    if unique_name is None:
        name = atom if isinstance(atom, str) else getattr(atom, 'name', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} FCC Lattice')
    if debye_temperature is None:
        global_object.log.warning(f"Debye temperature not provided for '{unique_name}'. Setting to default value of 300 K.")
        debye_temperature = 300.0
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=[AtomSite(atom, *xyz, debye_temperature=debye_temperature) for xyz in _FCC],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def diamond_cubic(
    length_a: numbers.Real,
    atom: Atoms | str,
    debye_temperature: numbers.Real | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
) -> Lattice:
    """
    Create a diamond cubic lattice.

    This function creates a [`Lattice`][...Lattice] object with lattice constants $a=b=c$ and lattice angles
    $\\alpha=\\beta=\\gamma=90^\\circ$. The diamond lattice corresponds to two interpenetrating FCC lattices, offset along the
    body diagonal by one-quarter of the cube's length. The [`Lattice`][...Lattice] object is therefore populated with atoms at
    the fractional coordinates:

    1. $(0,0,0)$, $(\\frac{1}{2},\\frac{1}{2},0)$, $(\\frac{1}{2},0,\\frac{1}{2})$ and $(0,\\frac{1}{2},\\frac{1}{2})$.
    2. $(\\frac{1}{4},\\frac{1}{4},\\frac{1}{4})$, $(\\frac{3}{4},\\frac{3}{4},\\frac{1}{4})$, $(\\frac{3}{4},\\frac{1}{4},
    \\frac{3}{4})$ and $(\\frac{1}{4},\\frac{3}{4},\\frac{3}{4})$.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in Å.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. <br> If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'Diamond Cubic Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to the [`unique_name`][...Lattice.unique_name] if not
        provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the diamond cubic lattice.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If `atom` is not an [`Atoms`][...atoms.Atoms] enum member or a string.
    KeyError
        If the provided atom string does not correspond to a valid [`Atoms`][...atoms.Atoms] enum member.
    """
    if unique_name is None:
        name = atom if isinstance(atom, str) else getattr(atom, 'name', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} Diamond Cubic Lattice')
    if debye_temperature is None:
        global_object.log.warning(f"Debye temperature not provided for '{unique_name}'. Setting to default value of 300 K.")
        debye_temperature = 300.0
    sites = []
    for a, b, c in _FCC:
        sites.append(AtomSite(atom=atom, fract_a=a, fract_b=b, fract_c=c, debye_temperature=debye_temperature))
        sites.append(
            AtomSite(atom=atom, fract_a=a + 0.25, fract_b=b + 0.25, fract_c=c + 0.25, debye_temperature=debye_temperature)
        )
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=sites,
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def zincblende(
    length_a: numbers.Real,
    atom1: Atoms | str,
    atom2: Atoms | str,
    debye_temperature1: numbers.Real | None = None,
    debye_temperature2: numbers.Real | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
) -> Lattice:
    """
    Create a zincblende lattice.

    This function creates a [`Lattice`][...Lattice] object with lattice constants $a=b=c$ and lattice angles
    $\\alpha=\\beta=\\gamma=90^\\circ$. The zincblende lattice corresponds to a [`diamond_cubic`][..diamond_cubic] lattice with
    two different atomic species for the two FCC sublattices and as such is populated with atoms at the same fractional
    coordinates:

    1. $(0,0,0)$, $(\\frac{1}{2},\\frac{1}{2},0)$, $(\\frac{1}{2},0,\\frac{1}{2})$ and $(0,\\frac{1}{2},\\frac{1}{2})$.
    2. $(\\frac{1}{4},\\frac{1}{4},\\frac{1}{4})$, $(\\frac{3}{4},\\frac{3}{4},\\frac{1}{4})$, $(\\frac{3}{4},\\frac{1}{4},
    \\frac{3}{4})$ and $(\\frac{1}{4},\\frac{3}{4},\\frac{3}{4})$.

    Parameters
    ----------
    length_a : float | int
        The length of the cubic lattice vectors in Å.
    atom1 : Atoms | str
        The atomic species of the first sublattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    atom2 : Atoms | str
        The atomic species of the second sublattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    debye_temperature1 : float | int | None
        The Debye temperature of the atoms in the first sublattice in Kelvin. <br> If None, a default value of 300 K is used.
    debye_temperature2 : float | int | None
        The Debye temperature of the atoms in the second sublattice in Kelvin. <br> If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'Zincblende Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to the [`unique_name`][...Lattice.unique_name] if not
        provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the zincblende lattice.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If `atom1` or `atom2` is not an [`Atoms`][...atoms.Atoms] enum member or a string.
    KeyError
        If the provided `atom1` or `atom2` strings does not correspond to a valid [`Atoms`][...atoms.Atoms] enum member.
    """
    if unique_name is None:
        name1 = atom1 if isinstance(atom1, str) else getattr(atom1, 'name', 'Unknown')
        name2 = atom2 if isinstance(atom2, str) else getattr(atom2, 'name', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name1}{name2} Zincblende Lattice')
    if debye_temperature1 is None:
        global_object.log.warning(
            f"Debye temperature not provided for '{unique_name}' (atom1). Setting to default value of 300 K."
        )
        debye_temperature1 = 300.0
    if debye_temperature2 is None:
        global_object.log.warning(
            f"Debye temperature not provided for '{unique_name}' (atom2). Setting to default value of 300 K."
        )
        debye_temperature2 = 300.0
    sites = []
    for a, b, c in _FCC:
        sites.append(AtomSite(atom=atom1, fract_a=a, fract_b=b, fract_c=c, debye_temperature=debye_temperature1))
        sites.append(
            AtomSite(atom=atom2, fract_a=a + 0.25, fract_b=b + 0.25, fract_c=c + 0.25, debye_temperature=debye_temperature2)
        )
    lattice = Lattice.cubic(
        length_a=length_a,
        atom_sites=sites,
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice


def hexagonal_close_packed(
    length_a: numbers.Real,
    length_c: numbers.Real,
    atom: Atoms | str,
    debye_temperature: numbers.Real | None = None,
    unique_name: str | None = None,
    display_name: str | None = None,
) -> Lattice:
    """
    Create a hexagonal close-packed (HCP) lattice.

    This function creates a [`Lattice`][...Lattice] object with lattice constants $a=b\\neq c$ and lattice angles
    $\\alpha=\\beta=90^\\circ$, $\\gamma=120^\\circ$. The [`Lattice`][...Lattice] object is then populated with two
    atoms at the fractional coordinates $(0,0,0)$ and $(\\frac{2}{3},\\frac{1}{3},\\frac{1}{2})$, corresponding to the two
    atoms in the primitive HCP unit cell.

    Parameters
    ----------
    length_a : float | int
        The length of the a-axis lattice vector in Å.
    length_c : float | int
        The length of the c-axis lattice vector in Å.
    atom : Atoms | str
        The atomic species of the lattice, either an [`Atoms`][...atoms.Atoms] object or its name as a string.
    debye_temperature : float | int | None
        The Debye temperature of the atoms in Kelvin. <br> If None, a default value of 300 K is used.
    unique_name : str | None
        A unique identifier for the [`Lattice`][..]. Defaults to ``'HCP Lattice'`` appended by a unique integer.
    display_name : str | None
        A prettily formatted name for the [`Lattice`][..]. Defaults to the [`unique_name`][...Lattice.unique_name] if not
        provided.
    Returns
    -------
    Lattice
        A [`Lattice`][..] object representing the HCP lattice.

    Raises
    ------
    ValueError
        If ``length_a`` is not positive.
    TypeError
        If `atom` is not an [`Atoms`][...atoms.Atoms] enum member or a string.
    KeyError
        If the provided atom string does not correspond to a valid [`Atoms`][...atoms.Atoms] enum member.
    """
    if unique_name is None:
        name = atom if isinstance(atom, str) else getattr(atom, 'name', 'Unknown')
        unique_name = global_object.generate_unique_name(f'{name} HCP Lattice')
    if debye_temperature is None:
        global_object.log.warning(f"Debye temperature not provided for '{unique_name}'. Setting to default value of 300 K.")
        debye_temperature = 300.0
    lattice = Lattice.hexagonal(
        length_a=length_a,
        length_c=length_c,
        atom_sites=[AtomSite(atom, *xyz, debye_temperature=debye_temperature) for xyz in _HCP],
        unique_name=unique_name,
        display_name=display_name,
    )
    return lattice
