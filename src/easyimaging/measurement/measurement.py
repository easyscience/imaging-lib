#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipp as sc
from easyscience.base_classes import NewBase
from scipp import UnitError
from scitiff import load_scitiff

if TYPE_CHECKING:
    pass

class Measurement(NewBase):
    """
    Class responsible for managing the measurement data of a time-of-flight neutron imaging experiment.
    """
    def __init__(
        self, 
        data_array: sc.DataArray,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        
        if not isinstance(data_array, sc.DataArray):
            raise TypeError("data_array must be an instance of scipp.DataArray.")
        
        self._validate_data_array_coordinate(
            data_array, 'tof', 'time-of-flight information', 't', 'time', 's')
        
        if any(data_array.coords['tof'].to(unit='s') < sc.scalar(0, unit='s')):
            raise ValueError("time_of_flight values must be non-negative.")
        
        if 'x' in data_array.coords:
            self._validate_data_array_coordinate(
                data_array, 'x', 'pixels', 'x', 'length', 'm')
            self._pixel_positions_x = data_array.coords['x']
        elif 'x' not in data_array.dims:
            raise ValueError("data array must have an 'x' dimension.")
            
        if 'y' in data_array.coords:
            self._validate_data_array_coordinate(
                data_array, 'y', 'pixels', 'y', 'length', 'm')
            self._pixel_positions_y = data_array.coords['y']
        elif 'y' not in data_array.dims:
            raise ValueError("data array must have an 'y' dimension.")

        self._time_of_flight = data_array.coords['tof']

        super().__init__(unique_name=unique_name, display_name=display_name)

        self._data_array = data_array.copy(deep=False)

        self._regions_of_interest = []

    @classmethod
    def from_scitiff(cls, filename: str | Path, unique_name: str | None = None, display_name: str | None = None) -> Measurement:  # noqa: E501
        """
        Create a Measurement instance by loading data from a SciTIFF file.

        Parameters
        ----------
        filename : str | Path
            Path to the SciTIFF file.
        unique_name : str | None
            Unique identifier for the measurement.
        display_name : str | None
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
    
    @classmethod
    def from_tiff_stack(
        cls, 
        filename: str | Path, 
        time_of_flight: sc.Variable | np.array,
        x_positions: sc.Variable | np.array | None = None,
        y_positions: sc.Variable | np.array | None = None,
        unique_name: str | None = None, 
        display_name: str | None = None
        ) -> Measurement:
        """
        Create a Measurement instance by loading data from a TIFF stack file.
        
        Parameters
        ----------
        filename : str | Path
            Path to the TIFF stack file.
        time_of_flight : sc.Variable | np.array
            Array of time-of-flight values corresponding to the frames in the TIFF stack. 
            If a numpy array is provided, the unit is assumed to be seconds.
        x_positions : sc.Variable | np.array | None
            Array of x-coordinate positions for the pixels.
            If a numpy array is provided, the unit is assumed to be meters.
        y_positions : sc.Variable | np.array | None
            Array of y-coordinate positions for the pixels.
            If a numpy array is provided, the unit is assumed to be meters.
        unique_name : str | None
            Unique identifier for the measurement.
        display_name : str | None
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
            raise FileNotFoundError(f"Failed to load TIFF stack file '{filename}': {e}") from e
        
        try:
            data_array = data_array.rename_dims({'dim_0': 't', 'dim_1': 'y', 'dim_2': 'x'})
        except Exception as e:
            raise RuntimeError(f"Failed to rename dimensions for TIFF stack file '{filename}': {e}") from e

        time_of_flight = cls._validate_provided_coord(
            data_array, time_of_flight, 'time_of_flight', 't', 'frames in the TIFF stack', 'time', 's')
        data_array.coords['tof'] = time_of_flight
        
        if x_positions is not None:
            x_positions = cls._validate_provided_coord(
                data_array, x_positions, 'x_positions', 'x', 'pixels in the x dimension', 'length', 'm')
            data_array.coords['x'] = x_positions

        if y_positions is not None:
            y_positions = cls._validate_provided_coord(
                data_array, y_positions, 'y_positions', 'y', 'pixels in the y dimension', 'length', 'm')
            data_array.coords['y'] = y_positions

        instance = cls(data_array=data_array, unique_name=unique_name, display_name=display_name)
        return instance
        

    def _validate_data_array_coordinate(
            self,
            data_array : sc.DataArray, 
            coord_name: str,
            coord_context: str, 
            expected_dim: str,
            expected_dim_string: str, 
            expected_unit: str
            ) -> None:
        if data_array.coords.get(coord_name) is None or data_array.coords[coord_name].shape == ():
            raise ValueError(f"data array must contain '{coord_name}' coordinate for {coord_context}.")
        if data_array.coords[coord_name].dim != expected_dim:
            raise ValueError(f"'{coord_name}' coordinate must be of dimension '{expected_dim}'.")
        try:
            data_array.coords[coord_name].to(unit=expected_unit)
        except UnitError:
            raise UnitError(f"'{coord_name}' coordinate must have a unit of {expected_dim_string}, such as ('{expected_unit}').") from None  # noqa: E501
        
    @staticmethod
    def _validate_provided_coord(
            data_array: sc.DataArray,
            coord: sc.Variable | np.array,
            coord_name: str,
            dim: str,
            length_context: str,
            expected_dim_string: str,
            expected_unit: str
            ) -> sc.Variable:
        if not isinstance(coord, (sc.Variable, np.ndarray)):
            raise TypeError(f"{coord_name} must be a scipp Variable or a numpy Array.")
        if not len(coord) == data_array.sizes[dim]:
            raise ValueError(f"Length of {coord_name} array does not match the number of {length_context}.")
        if isinstance(coord, np.ndarray):
            coord = sc.array(dims=[dim], values=coord, unit=expected_unit)
        try:
            coord.to(unit=expected_unit)
        except UnitError:
            raise UnitError(f"{coord_name} must have a unit of {expected_dim_string}, such as '{expected_unit}'.") from None
        return coord
        
        