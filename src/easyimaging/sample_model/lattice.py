# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import Parameter
from easyscience import global_object
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
            temperature: Numeric = 300.0,
            unique_name: str | None = None,
            display_name: str | None = None
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
        temperature : float | int
            The temperature of the lattice in Kelvin.
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
        self._temperature = self._create_temperature_parameter(temperature)

    @classmethod
    def cubic(cls,
              length_a: Numeric,
              atom_sites: list[AtomSite] | None = None,
              temperature: Numeric = 300.0,
              unique_name: str | None = None,
              display_name: str | None = None
              ):
        """
        Create a cubic lattice with equal lengths and 90-degree angles.

        Parameters
        ----------
        length_a : float | int
            The length of the cubic lattice vectors in angstrom.
        atom_sites : list[AtomSite] | None
            A list of [`AtomSite`][..] objects to insert into the lattice.
        temperature : float | int
            The temperature of the lattice in Kelvin.
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to ``'CubicLattice'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Returns
        -------
        Lattice
            A new instance of a cubic [`Lattice`][..].
        """
        if unique_name is None:
            unique_name = global_object.generate_unique_name('CubicLattice')
        lattice = cls(length_a=length_a, length_b=length_a, length_c=length_a,
                   alpha=90.0, beta=90.0, gamma=90.0,
                   atom_sites=atom_sites,
                   temperature=temperature,
                   unique_name=unique_name,
                   display_name=display_name
                  )
        lattice._default_unique_name = True  # This gets set to False by the super init
        lattice.length_b.make_dependent_on('length_a', {'length_a': lattice.length_a})
        lattice.length_c.make_dependent_on('length_a', {'length_a': lattice.length_a})
        return lattice

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

    @property
    def temperature(self) -> Parameter:
        return self._temperature

    @temperature.setter
    def temperature(self, value: Numeric):
        if value < 0:
            raise ValueError("Temperature must be non-negative.")
        self._temperature.value = value

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

    def _create_temperature_parameter(self, temperature_value: Numeric) -> Parameter:
        unique_name = generate_unique_name_no_zero(f'{self.unique_name}_temperature')
        parameter = Parameter(value=temperature_value, unit='K', min=0.0, fixed=True, unique_name=unique_name)
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter
