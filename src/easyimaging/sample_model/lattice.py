# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import EasyList
from easyscience.base_classes import ModelBase

from .atom_site import AtomSite

Numeric = int | float
INFINITESIMAL = 1e-3


class Lattice(ModelBase):
    """A Lattice represents the periodic arrangement of atoms in a crystal structure, defined by its lattice
    parameters (``length_a``, ``length_b``, ``length_c``, ``alpha``, ``beta``, ``gamma``) and a list of
    [`AtomSite`][.atom_site.AtomSite] objects occupying it.

    A [`Lattice`][.] can be created directly via the constructor, via the [`cubic`][.cubic] or [`hexagonal`][.hexagonal]
     convenience constructors for the corresponding lattice symmetries, or via one of the crystal-structure
     convenience functions in [`easyimaging.sample_model.crystals`][..crystals], which additionally populate the
     [`AtomSite`][.atom_site.AtomSite] objects for common crystal structures.

    Example
    -------
    Creating a [Lattice][.] instance manually:
    ```python
    from easyimaging.sample_model import AtomSite, Atoms, Lattice

    lattice = Lattice(
        length_a=2.87,
        length_b=2.87,
        length_c=2.87,
        atom_sites=[
            AtomSite(atom=Atoms.Fe, fract_x=0.0, fract_y=0.0, fract_z=0.0),
            AtomSite(atom=Atoms.Fe, fract_x=0.5, fract_y=0.5, fract_z=0.5),
        ],
    )
    ```
    """

    def __init__(
        self,
        length_a: Numeric,
        length_b: Numeric,
        length_c: Numeric,
        alpha: Numeric = 90.0,
        beta: Numeric = 90.0,
        gamma: Numeric = 90.0,
        atom_sites: list[AtomSite] | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Initialize a Lattice instance.

        Parameters
        ----------
        length_a : float | int
            The length of the lattice vector along the x-axis in angstrom.
        length_b : float | int
            The length of the lattice vector along the y-axis in angstrom.
        length_c : float | int
            The length of the lattice vector along the z-axis in angstrom.
        alpha : float | int
            The angle between the b and c lattice vectors (in degrees).
        beta : float | int
            The angle between the a and c lattice vectors (in degrees).
        gamma : float | int
            The angle between the a and b lattice vectors (in degrees).
        atom_sites : list[AtomSite] | None
            A list of [`AtomSite`][..] objects to insert into the lattice.
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to ``'Lattice'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Raises
        ------
        ValueError
            If any of ``length_a``, ``length_b``, or ``length_c`` is not positive.<br>
            If any of ``alpha``, ``beta``, or ``gamma`` is not between 0 and 180 degrees.
        TypeError
            If ``atom_sites`` is provided and is not a list of [`AtomSite`][.atom_site.AtomSite] objects.
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        for length in (length_a, length_b, length_c):
            if length <= INFINITESIMAL:
                raise ValueError('Lattice lengths must be positive and non-zero.')
        if not (
            INFINITESIMAL < alpha < 180 - INFINITESIMAL
            and INFINITESIMAL < beta < 180 - INFINITESIMAL
            and INFINITESIMAL < gamma < 180 - INFINITESIMAL
        ):
            raise ValueError('Lattice angles alpha, beta, and gamma must be between 0 and 180 degrees.')

        self._atom_sites = EasyList(
            protected_types=AtomSite, unique_name=global_object.generate_unique_name(f'{self.unique_name}_atom_sites')
        )
        self._atom_sites._default_unique_name = True  # This gets set to False by the super init

        if atom_sites is not None:
            if not isinstance(atom_sites, list) or not all(isinstance(site, AtomSite) for site in atom_sites):
                raise TypeError('atom_sites must be a list of AtomSite objects.')
            self._atom_sites.extend(atom_sites)

        self._length_a = self._create_length_parameter(length_a, 'a')
        self._length_b = self._create_length_parameter(length_b, 'b')
        self._length_c = self._create_length_parameter(length_c, 'c')
        self._alpha = self._create_angle_parameter(alpha, 'alpha')
        self._beta = self._create_angle_parameter(beta, 'beta')
        self._gamma = self._create_angle_parameter(gamma, 'gamma')

    @classmethod
    def cubic(
        cls,
        length_a: Numeric,
        atom_sites: list[AtomSite] | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Create a cubic lattice with equal lengths and 90-degree angles.

        Parameters
        ----------
        length_a : float | int
            The length of the cubic lattice vectors in angstrom.
        atom_sites : list[AtomSite] | None
            A list of [`AtomSite`][..] objects to insert into the lattice.
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to ``'CubicLattice'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Returns
        -------
        Lattice
            A new instance of a cubic [`Lattice`][..].

        Raises
        ------
        ValueError
            If ``length_a`` is not positive.
        TypeError
            If ``atom_sites`` is provided and is not a list of [`AtomSite`][.atom_site.AtomSite] objects.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name('CubicLattice')
        lattice = cls(
            length_a=length_a,
            length_b=length_a,
            length_c=length_a,
            alpha=90.0,
            beta=90.0,
            gamma=90.0,
            atom_sites=atom_sites,
            unique_name=unique_name,
            display_name=display_name,
        )
        lattice._default_unique_name = True  # This gets set to False by the super init
        lattice.length_b.make_dependent_on('length_a', {'length_a': lattice.length_a})
        lattice.length_c.make_dependent_on('length_a', {'length_a': lattice.length_a})
        return lattice

    @classmethod
    def hexagonal(
        cls,
        length_a: Numeric,
        length_c: Numeric,
        atom_sites: list[AtomSite] | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Create a hexagonal lattice with equal lengths for a and b, 90-degree angles for alpha and beta,
        and 120-degree angle for gamma.

        Parameters
        ----------
        length_a : float | int
            The length of the a and b lattice vectors in angstrom.
        length_c : float | int
            The length of the c lattice vector in angstrom.
        atom_sites : list[AtomSite] | None
            A list of [`AtomSite`][..] objects to insert into the lattice.
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to ``'HexagonalLattice'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Returns
        -------
        Lattice
            A new instance of a hexagonal [`Lattice`][..].

        Raises
        ------
        ValueError
            If ``length_a`` or ``length_c`` is not positive.
        TypeError
            If ``atom_sites`` is provided and is not a list of [`AtomSite`][.atom_site.AtomSite] objects.
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name('HexagonalLattice')
        lattice = cls(
            length_a=length_a,
            length_b=length_a,
            length_c=length_c,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            atom_sites=atom_sites,
            unique_name=unique_name,
            display_name=display_name,
        )
        lattice._default_unique_name = True  # This gets set to False by the super init
        lattice.length_b.make_dependent_on('length_a', {'length_a': lattice.length_a})
        return lattice

    @property
    def length_a(self) -> Parameter:
        """The length of the lattice vector along the x-axis.

        Parameters
        ----------
        value : int | float
            A positive number, in angstrom.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the length of the lattice vector along the
            x-axis, in angstrom.

        Raises
        ------
        ValueError
            If ``value`` is not positive.
        """
        return self._length_a

    @length_a.setter
    def length_a(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if value <= INFINITESIMAL:
            raise ValueError('Lattice length must be positive and non-zero.')
        self._length_a.value = value

    @property
    def length_b(self) -> Parameter:
        """The length of the lattice vector along the y-axis.

        Parameters
        ----------
        value : int | float
            A positive number, in angstrom.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the length of the lattice vector along the
            y-axis, in angstrom.

        Raises
        ------
        ValueError
            If ``value`` is not positive.
        """
        return self._length_b

    @length_b.setter
    def length_b(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if value <= INFINITESIMAL:
            raise ValueError('Lattice length must be positive and non-zero.')
        self._length_b.value = value

    @property
    def length_c(self) -> Parameter:
        """The length of the lattice vector along the z-axis.

        Parameters
        ----------
        value : int | float
            A positive number, in angstrom.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the length of the lattice vector along the
            z-axis, in angstrom.

        Raises
        ------
        ValueError
            If ``value`` is not positive.
        """
        return self._length_c

    @length_c.setter
    def length_c(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if value <= INFINITESIMAL:
            raise ValueError('Lattice length must be positive and non-zero.')
        self._length_c.value = value

    @property
    def alpha(self) -> Parameter:
        """The angle between the b and c lattice vectors.

        Parameters
        ----------
        value : int | float
            A number between 0 and 180 degrees.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the angle between the b and c lattice vectors,
            in degrees.

        Raises
        ------
        ValueError
            If ``value`` is not between 0 and 180 degrees.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if not (INFINITESIMAL < value < 180 - INFINITESIMAL):
            raise ValueError('Lattice angle must be between 0 and 180 degrees.')
        self._alpha.value = value

    @property
    def beta(self) -> Parameter:
        """The angle between the a and c lattice vectors.

        Parameters
        ----------
        value : int | float
            A number between 0 and 180 degrees.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the angle between the a and c lattice vectors,
            in degrees.

        Raises
        ------
        ValueError
            If ``value`` is not between 0 and 180 degrees.
        """
        return self._beta

    @beta.setter
    def beta(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if not (INFINITESIMAL < value < 180 - INFINITESIMAL):
            raise ValueError('Lattice angle must be between 0 and 180 degrees.')
        self._beta.value = value

    @property
    def gamma(self) -> Parameter:
        """The angle between the a and b lattice vectors.

        Parameters
        ----------
        value : int | float
            A number between 0 and 180 degrees.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the angle between the a and b lattice vectors,
            in degrees.

        Raises
        ------
        ValueError
            If ``value`` is not between 0 and 180 degrees.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, value: Numeric):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if not (INFINITESIMAL < value < 180 - INFINITESIMAL):
            raise ValueError('Lattice angle must be between 0 and 180 degrees.')
        self._gamma.value = value

    @property
    def atom_sites(self) -> EasyList:
        """The atom sites contained in the lattice.

        Returns
        -------
        EasyList
            The [`EasyList`][easyscience.base_classes.EasyList] of [`AtomSite`][.atom_site.AtomSite] objects contained
            in the lattice.
        """
        return self._atom_sites

    def _create_length_parameter(self, length_value: Numeric, axis: str) -> Parameter:
        unique_name = global_object.generate_unique_name(f'{self.unique_name}_length_{axis}')
        parameter = Parameter(
            name=f'length_{axis}', value=length_value, unit='angstrom', min=INFINITESIMAL, fixed=True, unique_name=unique_name
        )
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter

    def _create_angle_parameter(self, angle_value: Numeric, axis: str) -> Parameter:
        unique_name = global_object.generate_unique_name(f'{self.unique_name}_angle_{axis}')
        parameter = Parameter(
            name=f'angle_{axis}',
            value=angle_value,
            unit='deg',
            min=INFINITESIMAL,
            max=180.0 - INFINITESIMAL,
            fixed=True,
            unique_name=unique_name,
        )
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter
