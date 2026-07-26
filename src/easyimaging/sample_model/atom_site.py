# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

from ..utils import generate_unique_name_no_zero

Numeric = int | float

KNOWN_SPECIES = [
 'H', 'H1', 'H2', 'H3', 'He', 'He3', 'He4', 'Li', 'Li6', 'Li7', 'Be', 'Be9', 'B', 'B10', 'B11', 'C', 'C12', 'C13', 'N', 'N14',
 'N15', 'O', 'O16', 'O17', 'O18', 'F', 'F19', 'Ne', 'Ne20', 'Ne21', 'Ne22', 'Na', 'Na23', 'Mg', 'Mg24', 'Mg25', 'Mg26', 'Al',
 'Al27', 'Si', 'Si28', 'Si29', 'Si30', 'P', 'P31', 'S', 'S32', 'S33', 'S34', 'S36', 'Cl', 'Cl35', 'Cl37', 'Ar', 'Ar36', 'Ar38',
 'Ar40', 'K', 'K39', 'K40', 'K41', 'Ca', 'Ca40', 'Ca42', 'Ca43', 'Ca44', 'Ca46', 'Ca48', 'Sc', 'Sc45', 'Ti', 'Ti46', 'Ti47',
 'Ti48', 'Ti49', 'Ti50', 'V', 'V50', 'V51', 'Cr', 'Cr50', 'Cr52', 'Cr53', 'Cr54', 'Mn', 'Mn55', 'Fe', 'Fe54', 'Fe56', 'Fe57',
 'Fe58', 'Co', 'Co59', 'Ni', 'Ni58', 'Ni60', 'Ni61', 'Ni62', 'Ni64', 'Cu', 'Cu63', 'Cu65', 'Zn', 'Zn64', 'Zn66', 'Zn67',
 'Zn68', 'Zn70', 'Ga', 'Ga69', 'Ga71', 'Ge', 'Ge70', 'Ge72', 'Ge73', 'Ge74', 'Ge76', 'As', 'As75', 'Se', 'Se74', 'Se76',
 'Se77', 'Se78', 'Se80', 'Se82', 'Br', 'Br79', 'Br81', 'Kr', 'Kr86', 'Rb', 'Rb85', 'Rb87', 'Sr', 'Sr84', 'Sr86', 'Sr87',
 'Sr88', 'Y', 'Y89', 'Zr', 'Zr90', 'Zr91', 'Zr92', 'Zr94', 'Zr96', 'Nb', 'Nb93', 'Mo', 'Mo92', 'Mo94', 'Mo95', 'Mo96', 'Mo97',
 'Mo98', 'Mo100', 'Tc', 'Ru', 'Rh', 'Rh103', 'Pd', 'Pd102', 'Pd104', 'Pd105', 'Pd106', 'Pd108', 'Pd110', 'Ag', 'Ag107',
 'Ag109', 'Cd', 'Cd106', 'Cd108', 'Cd110', 'Cd111', 'Cd112', 'Cd113', 'Cd114', 'Cd116', 'In', 'In113', 'In115', 'Sn',
 'Sn112', 'Sn114', 'Sn115', 'Sn116', 'Sn117', 'Sn118', 'Sn119', 'Sn120', 'Sn122', 'Sn124', 'Sb', 'Sb121', 'Sb123', 'Te',
 'Te120', 'Te122', 'Te123', 'Te124', 'Te125', 'Te126', 'Te128', 'Te130', 'I', 'I127', 'Xe', 'Cs', 'Cs133', 'Ba', 'Ba130',
 'Ba132', 'Ba134', 'Ba135', 'Ba136', 'Ba137', 'Ba138', 'La', 'La138', 'La139', 'Ce', 'Ce136', 'Ce138', 'Ce140', 'Ce142', 'Pr',
 'Pr141', 'Nd', 'Nd142', 'Nd143', 'Nd144', 'Nd145', 'Nd146', 'Nd148', 'Nd150', 'Pm', 'Sm', 'Sm144', 'Sm147', 'Sm148', 'Sm149',
 'Sm150', 'Sm152', 'Sm154', 'Eu', 'Eu151', 'Eu153', 'Gd', 'Gd152', 'Gd154', 'Gd155', 'Gd156', 'Gd157', 'Gd158', 'Gd160', 'Tb',
 'Tb159', 'Dy', 'Dy156', 'Dy158', 'Dy160', 'Dy161', 'Dy162', 'Dy163', 'Dy164', 'Ho', 'Ho165', 'Er', 'Er162', 'Er164', 'Er166',
 'Er167', 'Er168', 'Er170', 'Tm', 'Tm169', 'Yb', 'Yb168', 'Yb170', 'Yb171', 'Yb172', 'Yb173', 'Yb174', 'Yb176', 'Lu', 'Lu175',
 'Lu176', 'Hf', 'Hf174', 'Hf176', 'Hf177', 'Hf178', 'Hf179', 'Hf180', 'Ta', 'Ta180', 'Ta181', 'W', 'W180', 'W182', 'W183',
 'W184', 'W186', 'Re', 'Re185', 'Re187', 'Os', 'Os184', 'Os186', 'Os187', 'Os188', 'Os189', 'Os190', 'Os192', 'Ir', 'Pt',
 'Pt190', 'Pt192', 'Pt194', 'Pt195', 'Pt196', 'Pt198', 'Au', 'Au197', 'Hg', 'Hg196', 'Hg199', 'Tl', 'Tl203', 'Tl205', 'Pb',
 'Pb204', 'Pb206', 'Pb207', 'Pb208', 'Bi', 'Bi209', 'Ra', 'Th', 'Th232', 'Pa', 'U', 'U233', 'U234', 'U235', 'U238'
 ]


class AtomSite(ModelBase):
    """
    An AtomSite represents a specific position in a crystal structure and its occupying atom, defined by its atomic
    species label and fractional coordinates (x, y, z) within the unit cell.
    """

    def __init__(
            self,
            atomic_species: str,
            fract_x: Numeric,
            fract_y: Numeric,
            fract_z: Numeric,
            debye_temperature: Numeric | None = None,
            unique_name: str | None = None,
            display_name: str | None = None,
    ):
        """
        Initialize an AtomSite instance.

        Parameters
        ----------
        atomic_species : str
            The atomic species of the site.
        fract_x : int | float
            The fractional x-coordinate of the site.
        fract_y : int | float
            The fractional y-coordinate of the site.
        fract_z : int | float
            The fractional z-coordinate of the site.
        unique_name : str | None
            A unique identifier for the [`AtomSite`][..]. Defaults to ``'AtomSite'``, prepended with the atomic species label,
             and appended with a unique integer.
        display_name : str | None
            A prettily formatted name for the [`AtomSite`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Raises
        ------
        TypeError
            If `atomic_species` is not a string.<br>
            If any of the fractional coordinates are not floats or ints.
        ValueError
            If `atomic_species` is not a valid element or isotope, or if any of the
        """
        self._validate_atomic_species(atomic_species)
        self._atomic_species = atomic_species

        self._validate_fract_value(fract_x, 'x')
        self._validate_fract_value(fract_y, 'y')
        self._validate_fract_value(fract_z, 'z')

        if unique_name is None:
            unique_name = global_object.generate_unique_name(f'{atomic_species} AtomSite')
            super().__init__(unique_name=unique_name, display_name=display_name)
            self._default_unique_name = True  # This gets set to False by the super init
        else:
            super().__init__(unique_name=unique_name, display_name=display_name)

        self._fract_x = self._generate_fract_parameter(fract_x, 'x')
        self._fract_y = self._generate_fract_parameter(fract_y, 'y')
        self._fract_z = self._generate_fract_parameter(fract_z, 'z')

        if debye_temperature is None:
            global_object.log.warning(f"Debye temperature not provided for AtomSite '{self.unique_name}'."
                                      "Setting to default value of 300 K.")
            debye_temperature = 300.0
        else:
            self._validate_debye_temperature(debye_temperature)
        debye_name = generate_unique_name_no_zero(f'{self.unique_name}_debye_temperature')
        self._debye_temperature = Parameter(value=debye_temperature, unit='K', min=0.0, fixed=True, unique_name=debye_name)

    @property
    def atomic_species(self) -> str:
        return self._atomic_species

    @atomic_species.setter
    def atomic_species(self, value: str):
        self._validate_atomic_species(value)
        self._atomic_species = value
        if self._default_unique_name:
            self.unique_name = global_object.generate_unique_name(f'{value} AtomSite')
            # Change _default_unique_name when Parameter uses NewBase
            self.fract_x.unique_name = generate_unique_name_no_zero(f'{self.unique_name}_fract_x')
            self.fract_y.unique_name = generate_unique_name_no_zero(f'{self.unique_name}_fract_y')
            self.fract_z.unique_name = generate_unique_name_no_zero(f'{self.unique_name}_fract_z')
            self.debye_temperature.unique_name = generate_unique_name_no_zero(f'{self.unique_name}_debye_temperature')

    @property
    def fract_x(self) -> Parameter:
        return self._fract_x

    @fract_x.setter
    def fract_x(self, value: Numeric):
        self._validate_fract_value(value, 'x')
        self._fract_x.value = value

    @property
    def fract_y(self) -> Parameter:
        return self._fract_y

    @fract_y.setter
    def fract_y(self, value: Numeric):
        self._validate_fract_value(value, 'y')
        self._fract_y.value = value

    @property
    def fract_z(self) -> Parameter:
        return self._fract_z

    @fract_z.setter
    def fract_z(self, value: Numeric):
        self._validate_fract_value(value, 'z')
        self._fract_z.value = value

    @property
    def debye_temperature(self) -> Parameter:
        return self._debye_temperature

    @debye_temperature.setter
    def debye_temperature(self, value: Numeric):
        self._validate_debye_temperature(value)
        self._debye_temperature.value = value

    @fract_z.setter
    def fract_z(self, value: Numeric):
        self._validate_fract_value(value, 'z')
        self._fract_z.value = value

    def _validate_atomic_species(self, value: str):
        if not isinstance(value, str):
            raise TypeError('"atomic_species" must be a string representing an element or isotope')
        if value not in KNOWN_SPECIES:
            raise ValueError(f'"atomic_species" must be a valid element or isotope. Got: {value}')

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

    def _generate_fract_parameter(self, fract_value: Numeric, axis: str) -> Parameter:

        unique_name = generate_unique_name_no_zero(f'{self.unique_name}_fract_{axis}')

        # Change _default_unique_name when Parameter uses NewBase
        return Parameter(value=fract_value, min=0.0, max=1.0, fixed=True, unique_name=unique_name)

    def __repr__(self):
        return (f"AtomSite(atomic_species='{self.atomic_species}', fract_x={self.fract_x.value},"
                f" fract_y={self.fract_y.value}, fract_z={self.fract_z.value})")
