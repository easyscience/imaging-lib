# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
from typing import Any

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

from .atoms import Atoms

DEBYE_INFINITESIMAL = 1e-9


class AtomSite(ModelBase):
    """An [`AtomSite`][.] object represents a specific position in a crystal structure and its occupying atom, defined by its
    [atomic species][...atoms.Atom], vibrational properties, and fractional coordinates $(a, b, c)$ within the host lattice's
    unit cell.

    Fractional coordinates $(a, b, c)$ are wrapped around the unit-cell to the half-open range [0, 1).

    An [`AtomSite`][.] is most commonly created as part of a [`Lattice`][...lattice.Lattice], either directly via its
    constructor, or indirectly through one of the crystal-structure convenience functions in
    [`easyimaging.sample_model.crystals`][...crystals].

    Example
    -------
    Creating an [`AtomSite`][.] instance manually:
    ```python
    from easyimaging.sample_model import AtomSite, Atoms

    site = AtomSite(
        atom=Atoms.Fe,
        fract_a=0.0,
        fract_b=0.0,
        fract_c=0.0,
        debye_temperature=470.0,
    )
    ```
    """

    def __init__(
        self,
        atom: Atoms | str,
        fract_a: numbers.Real,
        fract_b: numbers.Real,
        fract_c: numbers.Real,
        debye_temperature: numbers.Real | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Initialize an AtomSite instance.

        Parameters
        ----------
        atom : Atoms | str
            The atomic species that occupies the site. Can be either an [`Atoms`][....atoms.Atoms] object or its name as a
            string.
        fract_a : int | float
            The fractional a-coordinate of the site.
        fract_b : int | float
            The fractional b-coordinate of the site.
        fract_c : int | float
            The fractional c-coordinate of the site.
        debye_temperature : float | int | None
            The Debye temperature of the atom site in Kelvin. If None, a default value of 300 K is used.
        unique_name : str | None
            A unique identifier for the [`AtomSite`][..]. Defaults to ``'AtomSite'``, prepended with the atomic species label,
             and appended with a unique integer.
        display_name : str | None
            A prettily formatted name for the [`AtomSite`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Raises
        ------
        TypeError
            If `atom` is not an [`Atoms`][....atoms.Atoms] enum member or a string.<br>
            If any of the fractional coordinates or the debye_temperature are not numbers.
        ValueError
            If any of the fractional coordinates are not within the range [-1.0, 1.0].
            If the debye_temperature is not positive.
        KeyError
            If the provided atom string does not correspond to a valid [`Atoms`][....atoms.Atoms] enum member.
        """

        atom = self._validate_atom(atom)
        self._atom = atom

        wrapped_fract_a = self._validate_fract_value(fract_a, 'a')
        wrapped_fract_b = self._validate_fract_value(fract_b, 'b')
        wrapped_fract_c = self._validate_fract_value(fract_c, 'c')
        if debye_temperature is not None:
            self._validate_debye_temperature(debye_temperature)

        if unique_name is None:
            unique_name = global_object.generate_unique_name(f'{atom.name} AtomSite')
            super().__init__(unique_name=unique_name, display_name=display_name)
            self._default_unique_name = True  # This gets set to False by the super init
        else:
            super().__init__(unique_name=unique_name, display_name=display_name)

        if debye_temperature is None:
            global_object.log.warning(
                f"Debye temperature not provided for AtomSite '{self.unique_name}'. Setting to default value of 300 K."
            )
            debye_temperature = 300.0

        self._fract_a = self._generate_fract_parameter(wrapped_fract_a, 'a')
        self._fract_b = self._generate_fract_parameter(wrapped_fract_b, 'b')
        self._fract_c = self._generate_fract_parameter(wrapped_fract_c, 'c')

        debye_name = global_object.generate_unique_name(f'{self.unique_name}_debye_temperature')
        self._debye_temperature = Parameter(
            name='debye_temperature',
            value=debye_temperature,
            unit='K',
            min=DEBYE_INFINITESIMAL,
            fixed=True,
            unique_name=debye_name,
        )

    @property
    def atom(self) -> Atoms:
        """The atomic species occupying the site.

        Parameters
        ----------
        value : Atoms | str
            An [`Atoms`][....atoms.Atoms] enum member, or the name of one as a string.

        Returns
        -------
        Atoms
            The atomic species occupying the site.

        Raises
        ------
        TypeError
            If ``value`` is neither an [`Atoms`][....atoms.Atoms] enum member nor a string.
        KeyError
            If ``value`` is a string that does not correspond to a valid [`Atoms`][....atoms.Atoms] member.
        """
        return self._atom

    @atom.setter
    def atom(self, value: Atoms | str):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        value = self._validate_atom(value)
        self._atom = value
        if self._default_unique_name:
            self.unique_name = global_object.generate_unique_name(f'{value.name} AtomSite')
            self._default_unique_name = True
            # Change _default_unique_name when Parameter uses NewBase
            self.fract_a.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_a')
            self.fract_b.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_b')
            self.fract_c.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_c')
            self.debye_temperature.unique_name = global_object.generate_unique_name(f'{self.unique_name}_debye_temperature')

    @property
    def fract_a(self) -> Parameter:
        """The fractional x-coordinate of the site.

        The value is wrapped around the unit-cell to the half-open range [0, 1).

        Parameters
        ----------
        value : int | float
            A number between -1.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional x-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not between -1.0 and 1.0.
        """
        return self._fract_a

    @fract_a.setter
    def fract_a(self, value: numbers.Real):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        wrapped_value = self._validate_fract_value(value, 'a')
        self._fract_a.value = wrapped_value

    @property
    def fract_b(self) -> Parameter:
        """The fractional y-coordinate of the site.

        The value is wrapped around the unit-cell to the half-open range [0, 1).

        Parameters
        ----------
        value : int | float
            A number between -1.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional y-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not between -1.0 and 1.0.
        """
        return self._fract_b

    @fract_b.setter
    def fract_b(self, value: numbers.Real):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        wrapped_value = self._validate_fract_value(value, 'b')
        self._fract_b.value = wrapped_value

    @property
    def fract_c(self) -> Parameter:
        """The fractional z-coordinate of the site.

        The value is wrapped around the unit-cell to the half-open range [0, 1).

        Parameters
        ----------
        value : int | float
            A number between -1.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional z-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not between -1.0 and 1.0.
        """
        return self._fract_c

    @fract_c.setter
    def fract_c(self, value: numbers.Real):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        wrapped_value = self._validate_fract_value(value, 'c')
        self._fract_c.value = wrapped_value

    @property
    def debye_temperature(self) -> Parameter:
        """The Debye temperature of the atom site.

        Parameters
        ----------
        value : int | float
            A positive number, in Kelvin.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the Debye temperature of the atom site, in Kelvin.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not positive.
        """
        return self._debye_temperature

    @debye_temperature.setter
    def debye_temperature(self, value: numbers.Real):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._validate_debye_temperature(value)
        self._debye_temperature.value = value

    @property
    def unique_name(self) -> str:
        """
        The unique identifier of the AtomSite object.

        Parameters
        ----------
        value : str
            The new unique name for the [AtomSite][..] object.

        Returns
        -------
        str
            The current unique name of the [AtomSite][..].

        Raises
        ------
        TypeError
            If ``value`` is not a string.
        ValueError
            If the new unique name is already in use.

        Notes
        -----
        Setting a new unique name does not free up the old name.
        """
        return super().unique_name

    @unique_name.setter
    # Setters have no docstrings. They should be written in the getter docstring instead.
    def unique_name(self, value: str):
        super(AtomSite, type(self)).unique_name.fset(self, value)

    @property
    def display_name(self) -> str:
        """
        The pretty display name of the AtomSite object.

        This name is used when the object is displayed in user interfaces and reports.

        Parameters
        ----------
        value : str | None
            The new display name for the [AtomSite][..] object.

        Returns
        -------
        str
            The current display name of the [AtomSite][..].

        Raises
        ------
        TypeError
            If ``value`` is not a string or ``None``.

        Notes
        -----
        If it has not been set, the [unique name][..unique_name] of the [AtomSite][..] object
        will be used as its display name.
        """
        return super().display_name

    @display_name.setter
    # Setters have no docstrings. They should be written in the getter docstring instead.
    def display_name(self, value: str):
        super(AtomSite, type(self)).display_name.fset(self, value)

    def _validate_fract_value(self, value: numbers.Real, axis: str):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f'"fract_{axis}" must be a numeric value')
        if not (-1.0 <= value <= 1.0):
            raise ValueError(f'"fract_{axis}" must be between -1.0 and 1.0')
        wrapped_value = value % 1.0
        return 0.0 if wrapped_value == 1.0 else wrapped_value  # a tiny negative value rounds up to 1.0

    def _validate_debye_temperature(self, value: numbers.Real):
        if not isinstance(value, numbers.Real):
            raise TypeError('"debye_temperature" must be a numeric value')
        if value <= DEBYE_INFINITESIMAL:
            raise ValueError('"debye_temperature" must be positive')

    def _validate_atom(self, value):
        if isinstance(value, str):
            try:
                value = Atoms[value]
            except KeyError:
                raise KeyError(f'"atom" must be a valid Atoms enum or a valid Atoms name string. Got: {value}')
        elif not isinstance(value, Atoms):
            raise TypeError(f'"atom" must be a valid Atoms enum or a valid Atoms name string. Got: {value}')
        return value

    def _generate_fract_parameter(self, fract_value: numbers.Real, axis: str) -> Parameter:

        unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_{axis}')

        # Change _default_unique_name when Parameter uses NewBase
        return Parameter(name=f'fract_{axis}', value=fract_value, min=0.0, max=1.0, fixed=True, unique_name=unique_name)

    def __repr__(self):
        """Return a string representation of the AtomSite.

        Returns
        -------
        str
            A human-readable summary including the atomic species and fractional coordinates.
        """
        return (
            f"AtomSite(atomic_species='{self.atom.name}', fract_a={self.fract_a.value},"
            f' fract_b={self.fract_b.value}, fract_c={self.fract_c.value})'
        )

    def to_dict(self, skip: list[str] | None = None) -> dict[str, Any]:
        """Convert the AtomSite object to a dictionary representation.

        Used for saving the [AtomSite][..] to file.

        Parameters
        ----------
        skip : List[str] | None
            A list of attribute names to exclude from the output dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary representation of the [AtomSite][..] object.

        Notes
        -----
        A [unique_name][..unique_name] that was generated automatically is not
        written to the dictionary; only an explicitly supplied one is, so that a
        deserialized object gets a fresh name instead of clashing with an existing one.
        Likewise [display_name][..display_name] is omitted when it has not been set.
        """
        temp_dict = super().to_dict(skip=skip)
        if 'atom' not in (skip or []):
            temp_dict['atom'] = self.atom.name
        return temp_dict

    @classmethod
    def from_dict(cls, obj_dict: dict[str, Any]) -> AtomSite:
        """Recreate the AtomSite object from its dictionary representation created with the [to_dict][to_dict] method.

        Parameters
        ----------
        obj_dict : dict
            A dictionary representation of the [AtomSite][..] object.

        Returns
        -------
        AtomSite
            An instance of the [AtomSite][..] object recreated from the dictionary.

        Raises
        ------
        ValueError
            If the dictionary does not represent a serialized [AtomSite][..] object.
        """
        return super().from_dict(obj_dict)

    def __copy__(self) -> AtomSite:
        """Create a deep copy of the AtomSite object.

        Notes
        -----
        This uses the [to_dict][to_dict] and [from_dict][from_dict] methods to create the deep copy.

        Returns
        -------
        AtomSite
            A deep copy of the AtomSite object.
        """
        return super().__copy__()
