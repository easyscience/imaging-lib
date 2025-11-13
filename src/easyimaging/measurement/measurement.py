#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>

from easyscience.job.experiment import ExperimentBase
from scipp import DataArray
from scipp import UnitError


class Measurement(ExperimentBase):
    """
    Class responsible for managing the measurement data of a time-of-flight neutron imaging experiment.
    """
    def __init__(
        self, 
        name: str,
        data_array: DataArray
    ):
        if data_array.coords.get('tof') is None or data_array.coords['tof'].shape == ():
            raise ValueError("DataArray must contain 'tof' coordinate for time-of-flight information.")
        if data_array.coords['tof'].dim != 't':
            raise ValueError("'tof' coordinate must be of dimension time: 't'.")
        try:
            temp_tof_coords = data_array.coords['tof'].copy(deep=True)
            temp_tof_coords.to(unit='s')
        except UnitError:
            raise UnitError("'tof' coordinate must have a unit of time, such as seconds ('s').") from None

        if data_array.coords.get('x') is None or data_array.coords['x'].shape == ():
            raise ValueError("DataArray must contain 'x' coordinate for pixels.")
        if data_array.coords['x'].dim != 'x':
            raise ValueError("'x' coordinate must be of dimension 'x'.")
        try:
            temp_x_coords = data_array.coords['x'].copy(deep=True)
            temp_x_coords.to(unit='m')
        except UnitError:
            raise UnitError("'x' coordinate must have a unit of length, such as meters ('m').") from None

        if data_array.coords.get('y') is None or data_array.coords['y'].shape == ():
            raise ValueError("DataArray must contain 'y' coordinate for pixels.")
        if data_array.coords['y'].dim != 'y':
            raise ValueError("'y' coordinate must be of dimension 'y'.")
        try:
            temp_y_coords = data_array.coords['y'].copy(deep=True)
            temp_y_coords.to(unit='m')
        except UnitError:
            raise UnitError("'y' coordinate must have a unit of length, such as meters ('m').") from None

        self._pixel_positions_y = data_array.coords['y']

        super().__init__(name)

        self._data_array = data_array

        self._regions_of_interest = []

        