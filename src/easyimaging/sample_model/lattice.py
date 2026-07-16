# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import Parameter
from easyscience.base_classes import ModelBase
from easysience.base_classes import EasyList

from ..utils import generate_unique_name_no_zero
from .atom_site import AtomSite

Numeric = int | float


class Lattice(ModelBase):
    """
    A Lattice represents the periodic arrangement of atoms in a crystal structure, defined by its lattice parameters.
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
            display_name: str | None = None
            ):
        """
        Initialize a Lattice instance.

        Parameters
        ----------
        length_a : float | int
            The length of the lattice vector along the x-axis.
        length_b : float | int
            The length of the lattice vector along the y-axis.
        length_c : float | int
            The length of the lattice vector along the z-axis.
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
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        for length in (length_a, length_b, length_c):
            if length <= 0:
                raise ValueError("Lattice lengths must be positive.")
        if not (0 < alpha < 180 and 0 < beta < 180 and 0 < gamma < 180):
            raise ValueError("Lattice angles alpha, beta, and gamma must be between 0 and 180 degrees.")

        self._atom_sites = EasyList(
            protected_types=AtomSite,
            unique_name=generate_unique_name_no_zero(f'{self.unique_name}_atom_sites')
            )
        self._atom_sites._default_unique_name = True  # This gets set to False by the super init
        self._debye_temperatures = EasyList(
            protected_types=Parameter,
            unique_name=generate_unique_name_no_zero(f'{self.unique_name}_debye_temperatures')
        )
        self._debye_temperatures._default_unique_name = True  # This gets set to False by the super init

        if atom_sites is not None:
            if not isinstance(atom_sites, list) and not all(isinstance(site, AtomSite) for site in atom_sites):
                raise TypeError("atom_sites must be a list of AtomSite objects.")
            self._atom_sites.extend(atom_sites)

        self._length_a = self._create_length_parameter(length_a, 'a')
        self._length_b = self._create_length_parameter(length_b, 'b')
        self._length_c = self._create_length_parameter(length_c, 'c')
        self._alpha = self._create_angle_parameter(alpha, 'alpha')
        self._beta = self._create_angle_parameter(beta, 'beta')
        self._gamma = self._create_angle_parameter(gamma, 'gamma')

    @property
    def length_a(self) -> Parameter:
        return self._length_a

    @length_a.setter
    def length_a(self, value: Numeric):
        if value <= 0:
            raise ValueError("Lattice length must be positive.")
        self._length_a.value = value

    @property
    def length_b(self) -> Parameter:
        return self._length_b

    @length_b.setter
    def length_b(self, value: Numeric):
        if value <= 0:
            raise ValueError("Lattice length must be positive.")
        self._length_b.value = value

    @property
    def length_c(self) -> Parameter:
        return self._length_c

    @length_c.setter
    def length_c(self, value: Numeric):
        if value <= 0:
            raise ValueError("Lattice length must be positive.")
        self._length_c.value = value

    @property
    def alpha(self) -> Parameter:
        return self._alpha

    @alpha.setter
    def alpha(self, value: Numeric):
        if not (0 < value < 180):
            raise ValueError("Lattice angle must be between 0 and 180 degrees.")
        self._alpha.value = value

    @property
    def beta(self) -> Parameter:
        return self._beta

    @beta.setter
    def beta(self, value: Numeric):
        if not (0 < value < 180):
            raise ValueError("Lattice angle must be between 0 and 180 degrees.")
        self._beta.value = value

    @property
    def gamma(self) -> Parameter:
        return self._gamma

    @gamma.setter
    def gamma(self, value: Numeric):
        if not (0 < value < 180):
            raise ValueError("Lattice angle must be between 0 and 180 degrees.")
        self._gamma.value = value

    @property
    def atom_sites(self) -> EasyList:
        return self._atom_sites

    def add_atom_site(
            self,
            atomic_species: str,
            fract_x: Numeric,
            fract_y: Numeric,
            fract_z: Numeric,
            unique_name: str | None = None,
            display_name: str | None = None
            ):
        """
        Create a new AtomSite and add it to the lattice.

        Parameters
        ----------
        atomic_species : str
            The atomic species of the atom site.
        fract_x : float | int
            The fractional x-coordinate of the atom site.
        fract_y : float | int
            The fractional y-coordinate of the atom site.
        fract_z : float | int
            The fractional z-coordinate of the atom site.
        unique_name : str | None
            A unique identifier for the [`AtomSite`][..]. Defaults to ``'AtomSite'``, prepended with the atomic species label,
            and appended with a unique integer.
        display_name : str | None
            A prettily formatted name for the [`AtomSite`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
        """
        new_atom_site = AtomSite(
            atomic_species=atomic_species,
            fract_x=fract_x,
            fract_y=fract_y,
            fract_z=fract_z,
            unique_name=unique_name,
            display_name=display_name
        )
        self._atom_sites.append(new_atom_site)

    def _create_length_parameter(self, length_value: Numeric, axis: str) -> Parameter:
        unique_name = generate_unique_name_no_zero(f'{self.unique_name}_length_{axis}')
        parameter = Parameter(value=length_value, unit='angstrom', min=0.0, fixed=True, unique_name=unique_name)
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter

    def _create_angle_parameter(self, angle_value: Numeric, axis: str) -> Parameter:
        unique_name = generate_unique_name_no_zero(f'{self.unique_name}_angle_{axis}')
        parameter = Parameter(value=angle_value, min=0.0, max=180.0, fixed=True, unique_name=unique_name)
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter
