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
    parameters, lattice constants: (``length_a``, ``length_b``, ``length_c``) and angles: (``alpha``, ``beta``, ``gamma``)
    and a list of [`AtomSite`][..atom_site.AtomSite] objects, with their corresponding [atomic species][..atoms.Atoms],
    occupying it.

    A [`Lattice`][.] object can currently be created in 3 ways of descending specificity:

    - Directly via the constructor by supplying all lattice parameters and atomic sites.
    - Using the [`cubic`][.cubic] or [`hexagonal`][.hexagonal] convenience methods for the corresponding lattice symmetries.
    Atomic sites still needs to be supplied manually.
    - With one of the crystal-structure convenience functions in the [`crystals`][..crystals] module, which
    additionally populates the [`AtomSite`][..atom_site.AtomSite] objects for common crystal structures.

    Example
    -------
    Creating a hexagonal close-packed (HCP) magnesium [Lattice][.]:

    **1. Directly via the constructor**

    ```python
    from easyimaging.sample_model import AtomSite, Atoms, Lattice

    lattice = Lattice(
        length_a=3.21,
        length_b=3.21,
        length_c=5.21,
        alpha=90.0,
        beta=90.0,
        gamma=120.0,
        atom_sites=[
            AtomSite(atom=Atoms.Mg, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=400.0),
            AtomSite(atom=Atoms.Mg, fract_x=2 / 3, fract_y=1 / 3, fract_z=0.5, debye_temperature=400.0),
        ],
    )
    ```

    **2. Using the [`hexagonal`][.hexagonal] convenience constructor**

    ```python
    from easyimaging.sample_model import AtomSite, Atoms, Lattice

    lattice = Lattice.hexagonal(
        length_a=3.21,
        length_c=5.21,
        atom_sites=[
            AtomSite(atom=Atoms.Mg, fract_x=0.0, fract_y=0.0, fract_z=0.0, debye_temperature=400.0),
            AtomSite(atom=Atoms.Mg, fract_x=2 / 3, fract_y=1 / 3, fract_z=0.5, debye_temperature=400.0),
        ],
    )
    ```

    **3. Using the [`hexagonal_close_packed`][..crystals.hexagonal_close_packed] constructor from the
    [`crystals`][..crystals] module**

    ```python
    from easyimaging.sample_model import Atoms
    from easyimaging.sample_model.crystals import hexagonal_close_packed

    lattice = hexagonal_close_packed(
        length_a=3.21,
        length_c=5.21,
        atom=Atoms.Mg,
        debye_temperature=400.0,
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
            The lattice constant along the x-axis in Å.
        length_b : float | int
            The lattice constant along the y-axis in Å.
        length_c : float | int
            The lattice constant along the z-axis in Å.
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
            If any of ``length_a``, ``length_b``, or ``length_c`` is not positive and non-zero.<br>
            If any of ``alpha``, ``beta``, or ``gamma`` is not between 0 and 180 degrees.
        TypeError
            If any of ``length_a``, ``length_b``, or ``length_c`` is not a numeric value.
            If any of ``alpha``, ``beta``, or ``gamma`` is not a numeric value.
            If ``atom_sites`` is provided and is not a list of [`AtomSite`][...atom_site.AtomSite] objects.
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        for length, axis in ((length_a, 'a'), (length_b, 'b'), (length_c, 'c')):
            self._validate_length(length, axis)
        for angle, name in ((alpha, 'alpha'), (beta, 'beta'), (gamma, 'gamma')):
            self._validate_angle(angle, name)

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
        self._validate_length(value, 'a')
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
        self._validate_length(value, 'b')
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
        self._validate_length(value, 'c')
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
        self._validate_angle(value, 'alpha')
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
        self._validate_angle(value, 'beta')
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
        self._validate_angle(value, 'gamma')
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

    def _validate_length(self, value: Numeric, axis: str):
        if not isinstance(value, Numeric):
            raise TypeError(f'Lattice length_{axis} must be a numeric value. Got {type(value).__name__}')
        if not (value > INFINITESIMAL):
            raise ValueError(f'Lattice length_{axis} must be positive and non-zero.')

    def _validate_angle(self, value: Numeric, name: str):
        if not isinstance(value, Numeric):
            raise TypeError(f'Lattice angle {name} must be a numeric value. Got {type(value).__name__}')
        if not (INFINITESIMAL < value < 180 - INFINITESIMAL):
            raise ValueError(f'Lattice angle {name} must be between 0 and 180 degrees.')

    def _create_length_parameter(self, length_value: Numeric, axis: str) -> Parameter:
        unique_name = global_object.generate_unique_name(f'{self.unique_name}_length_{axis}')
        parameter = Parameter(
            name=f'length_{axis}', value=length_value, unit='angstrom', min=INFINITESIMAL, fixed=True, unique_name=unique_name
        )
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter

    def _create_angle_parameter(self, angle_value: Numeric, name: str) -> Parameter:
        unique_name = global_object.generate_unique_name(f'{self.unique_name}_angle_{name}')
        parameter = Parameter(
            name=f'angle_{name}',
            value=angle_value,
            unit='deg',
            min=INFINITESIMAL,
            max=180.0 - INFINITESIMAL,
            fixed=True,
            unique_name=unique_name,
        )
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter
