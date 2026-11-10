# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numbers

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase

TEMPERATURE_INFINITESIMAL = 1e-9
THICKNESS_INFINITESIMAL = 1e-5


class SampleProperties(ModelBase):
    """
    A [`SampleProperties`][.] object represents the macroscopic physical properties of a sample to be modelled. Together with a
    [`Lattice`][...lattice.Lattice] object, which describes the atomic structure of the sample, this object defines the model
    of the sample.
    """

    def __init__(
        self,
        sample_thickness: numbers.Real,
        temperature: numbers.Real | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Initialize a SampleProperties object.

        Parameters
        ----------
        sample_thickness : int | float
            The thickness of the sample in cm.
        temperature : int | float | None
            The temperature of the sample in Kelvin. If not provided, defaults to 300 K.
        unique_name : str | None
            A unique identifier for the [`SampleProperties`][..]. Defaults to ``'SampleProperties'`` appended by a unique
            integer.
        display_name : str | None
            A prettily formatted name for the [`SampleProperties`][..]. Defaults to [`unique_name`][..unique_name] if not
            provided.

        Raises
        ------
        TypeError
            If `sample_thickness` or `temperature` is not a numeric value.
        ValueError
            If `sample_thickness` or `temperature` is not positive.
        """
        if temperature is not None:
            self._validate_temperature(temperature)
        self._validate_thickness(sample_thickness)
        super().__init__(unique_name=unique_name, display_name=display_name)
        if temperature is None:
            global_object.log.warning(
                f"Temperature not provided for SampleProperties '{self.unique_name}'. Setting to default value of 300 K."
            )
            temperature = 300.0
        self._temperature = Parameter(
            value=temperature,
            unit='K',
            min=TEMPERATURE_INFINITESIMAL,
            fixed=True,
            unique_name=global_object.generate_unique_name(f'{self.unique_name}_temperature'),
        )
        self._temperature._default_unique_name = True
        self._sample_thickness = Parameter(
            value=sample_thickness,
            unit='cm',
            min=THICKNESS_INFINITESIMAL,
            fixed=True,
            unique_name=global_object.generate_unique_name(f'{self.unique_name}_sample_thickness'),
        )
        self._sample_thickness._default_unique_name = True

    @property
    def temperature(self) -> Parameter:
        """The physical temperature of the sample.

        Parameters
        ----------
        value : int | float
            A positive number, in Kelvin.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the temperature of the sample, in Kelvin.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not positive.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: numbers.Real):
        self._validate_temperature(value)
        self._temperature.value = value

    @property
    def sample_thickness(self) -> Parameter:
        """The thickness of the sample.

        Parameters
        ----------
        value : int | float
            A positive number, in centimeters.

        Returns
        -------
        Parameter
            The [`Parameter`][easyscience.variable.Parameter] holding the thickness of the sample, in centimeters.

        Raises
        ------
        TypeError
            If ``value`` is not a number.
        ValueError
            If ``value`` is not positive.
        """
        return self._sample_thickness

    @sample_thickness.setter
    def sample_thickness(self, value: numbers.Real):
        self._validate_thickness(value)
        self._sample_thickness.value = value

    def _validate_temperature(self, value: numbers.Real):
        if not isinstance(value, numbers.Real):
            raise TypeError('"temperature" must be a numeric value')
        if value <= TEMPERATURE_INFINITESIMAL:
            raise ValueError('"temperature" must be positive')

    def _validate_thickness(self, value: numbers.Real):
        if not isinstance(value, numbers.Real):
            raise TypeError('"sample_thickness" must be a numeric value')
        if value <= THICKNESS_INFINITESIMAL:
            raise ValueError('"sample_thickness" must be positive')
