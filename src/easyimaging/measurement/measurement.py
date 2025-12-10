#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import scipp as sc
from easyscience.base_classes import NewBase
from scipp import UnitError
from scitiff import load_scitiff

if TYPE_CHECKING:
    from typing import Optional

class Measurement(NewBase):
    """
    Class responsible for managing the measurement data of a time-of-flight neutron imaging experiment.
    """
    def __init__(
        self, 
        data_array: sc.DataArray,
        unique_name: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        
        if not isinstance(data_array, sc.DataArray):
            raise TypeError("data_array must be an instance of scipp.DataArray.")

        self._validate_data_array_coordinate(
            data_array, 'tof', 'time-of-flight information', 't', 'time', 's')

        self._validate_data_array_coordinate(
            data_array, 'x', 'pixels', 'x', 'length', 'm')
        
        self._validate_data_array_coordinate(
            data_array, 'y', 'pixels', 'y', 'length', 'm')

        self._pixel_positions_y = data_array.coords['y']
        self._pixel_positions_x = data_array.coords['x']
        self._time_of_flight = data_array.coords['tof']

        super().__init__(unique_name=unique_name, display_name=display_name)

        self._data_array = data_array.copy(deep=False)

        self._regions_of_interest = []

    @classmethod
    def from_scitiff(cls, filename: str | Path, unique_name: Optional[str] = None, display_name: Optional[str] = None) -> Measurement:  # noqa: E501
        """
        Create a Measurement instance by loading data from a SciTIFF file.

        Parameters
        ----------
        filename : str
            Path to the SciTIFF file.
        unique_name : Optional[str]
            Unique identifier for the measurement.
        display_name : Optional[str]
            Display name for the measurement.

        Returns
        -------
        Measurement
            An instance of the Measurement class containing the loaded data.
        """
        if not isinstance(filename, (str, Path)):
            raise TypeError("filename must be a string or Path object.")
        try:
            data_array = load_scitiff(filename)['image']
        except Exception as e:
            raise RuntimeError(f"Failed to load SciTIFF file '{filename}': {e}") from e
        try:
            instance = cls(data_array=data_array, unique_name=unique_name, display_name=display_name)
        except Exception as e:
            raise RuntimeError(f"Tiff file '{filename}' not a proper SciTIFF file: {e}") from e
        return instance
    
    def _validate_data_array_coordinate(
            data_array : sc.DataArray, 
            coord_name: str,
            coord_context: str, 
            expected_dim: str,
            expected_dim_string: str, 
            expected_unit: str
            ) -> None:
        if data_array.coords.get(coord_name) is None or data_array.coords[coord_name].shape == ():
            raise ValueError(f"DataArray must contain '{coord_name}' coordinate for {coord_context}.")
        if data_array.coords[coord_name].dim != expected_dim:
            raise ValueError(f"'{coord_name}' coordinate must be of dimension '{expected_dim}'.")
        try:
            unit = data_array.coords[coord_name].unit
            temp_varable = sc.scalar(value=1, unit=unit)
            temp_varable.to(unit=expected_unit)
        except UnitError:
            raise UnitError(f"'{coord_name}' coordinate must have a unit of {expected_dim_string}, such as ('{expected_unit}').") from None  # noqa: E501
