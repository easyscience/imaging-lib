# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

from .atoms import Atoms

Numeric = int | float


class AtomSite(ModelBase):
    """An `AtomSite` object represents a specific position in a crystal structure and its occupying atom, defined by its
    atomic species, vibrational properties, and fractional coordinates (a, b, c) within the host lattice's unit cell.

    An [`AtomSite`][.] is most commonly created as part of a [`Lattice`][...lattice.Lattice], either directly via its
    constructor, or indirectly through one of the crystal-structure convenience functions in
    [`easyimaging.sample_model.crystals`][...crystals].

    Example
    -------
    Creating an [AtomSite][.] instance manually:
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
        fract_a: Numeric,
        fract_b: Numeric,
        fract_c: Numeric,
        debye_temperature: Numeric | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Initialize an AtomSite instance.

        Parameters
        ----------
        atom : Atoms | str
            The atomic species that occupies the site.
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
            If `atom` is not an Atoms enum.<br>
            If any of the fractional coordinates are not floats or ints.
        ValueError
            If any of the fractional coordinates are out of range.
        KeyError
            If the provided atom string does not correspond to a valid Atoms enum member.
        """

        atom = self._validate_atom(atom)
        self._atom = atom

        self._validate_fract_value(fract_a, 'a')
        self._validate_fract_value(fract_b, 'b')
        self._validate_fract_value(fract_c, 'c')

        if unique_name is None:
            unique_name = global_object.generate_unique_name(f'{atom._name_} AtomSite')
            super().__init__(unique_name=unique_name, display_name=display_name)
            self._default_unique_name = True  # This gets set to False by the super init
        else:
            super().__init__(unique_name=unique_name, display_name=display_name)

        self._fract_a = self._generate_fract_parameter(fract_a, 'a')
        self._fract_b = self._generate_fract_parameter(fract_b, 'b')
        self._fract_c = self._generate_fract_parameter(fract_c, 'c')

        if debye_temperature is None:
            global_object.log.warning(
                f"Debye temperature not provided for AtomSite '{self.unique_name}'.Setting to default value of 300 K."
            )
            debye_temperature = 300.0
        else:
            self._validate_debye_temperature(debye_temperature)
        debye_name = global_object.generate_unique_name(f'{self.unique_name}_debye_temperature')
        self._debye_temperature = Parameter(
            name='debye_temperature', value=debye_temperature, unit='K', min=0.0, fixed=True, unique_name=debye_name
        )

    @property
    def atom(self) -> Atoms:
        """The atomic species occupying the site.

        Parameters
        ----------
        value : Atoms | str
            An [`Atoms`][..atoms.Atoms] enum member, or the name of one as a string.

        Returns
        -------
        Atoms
            The atomic species occupying the site.

        Raises
        ------
        TypeError
            If ``value`` is neither an [`Atoms`][..atoms.Atoms] enum member nor a string.
        KeyError
            If ``value`` is a string that does not correspond to a valid [`Atoms`][..atoms.Atoms] member.
        """
        return self._atom

    @atom.setter
    def atom(self, value: Atoms | str):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        value = self._validate_atom(value)
        self._atom = value
        if self._default_unique_name:
            self.unique_name = global_object.generate_unique_name(f'{value._name_} AtomSite')
            # Change _default_unique_name when Parameter uses NewBase
            self.fract_a.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_a')
            self.fract_b.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_b')
            self.fract_c.unique_name = global_object.generate_unique_name(f'{self.unique_name}_fract_c')
            self.debye_temperature.unique_name = global_object.generate_unique_name(f'{self.unique_name}_debye_temperature')

    @property
    def fract_a(self) -> Parameter:
        """The fractional x-coordinate of the site.

        Parameters
        ----------
        value : int | float
            A number between 0.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional x-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is not between 0.0 and 1.0.
        """
        return self._fract_a

    @fract_a.setter
    def fract_a(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._validate_fract_value(value, 'a')
        self._fract_a.value = value

    @property
    def fract_b(self) -> Parameter:
        """The fractional y-coordinate of the site.

        Parameters
        ----------
        value : int | float
            A number between 0.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional y-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is not between 0.0 and 1.0.
        """
        return self._fract_b

    @fract_b.setter
    def fract_b(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._validate_fract_value(value, 'b')
        self._fract_b.value = value

    @property
    def fract_c(self) -> Parameter:
        """The fractional z-coordinate of the site.

        Parameters
        ----------
        value : int | float
            A number between 0.0 and 1.0.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the fractional z-coordinate of the site.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is not between 0.0 and 1.0.
        """
        return self._fract_c

    @fract_c.setter
    def fract_c(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._validate_fract_value(value, 'c')
        self._fract_c.value = value

    @property
    def debye_temperature(self) -> Parameter:
        """The Debye temperature of the atom site.

        Parameters
        ----------
        value : int | float
            A non-negative number, in Kelvin.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the Debye temperature of the atom site, in Kelvin.

        Raises
        ------
        TypeError
            If ``value`` is not numeric.
        ValueError
            If ``value`` is negative.
        """
        return self._debye_temperature

    @debye_temperature.setter
    def debye_temperature(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._validate_debye_temperature(value)
        self._debye_temperature.value = value

    def _validate_fract_value(self, value: Numeric, axis: str):
        if not isinstance(value, Numeric):
            raise TypeError(f'"fract_{axis}" must be a numeric value')
        if not (0.0 <= value <= 1.0):
            raise ValueError(f'"fract_{axis}" must be between 0.0 and 1.0')

    def _validate_debye_temperature(self, value: Numeric):
        if not isinstance(value, Numeric):
            raise TypeError('"debye_temperature" must be a numeric value')
        if value < 0.0:
            raise ValueError('"debye_temperature" must be non-negative')

    def _validate_atom(self, value):
        if isinstance(value, str):
            try:
                value = Atoms[value]
            except KeyError:
                raise KeyError(f'"atom" must be a valid Atoms enum or a valid Atoms name string. Got: {value}')
        elif value not in Atoms:
            raise TypeError(f'"atom" must be a valid Atoms enum or a valid Atoms name string. Got: {value}')
        return value

    def _generate_fract_parameter(self, fract_value: Numeric, axis: str) -> Parameter:

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
            f"AtomSite(atomic_species='{self.atom._name_}', fract_a={self.fract_a.value},"
            f' fract_b={self.fract_b.value}, fract_c={self.fract_c.value})'
        )
