#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ess.imaging as essimaging
import numpy as np
import scipp as sc
from easyscience.base_classes import NewBase
from scipp import UnitError
from scitiff import load_scitiff

Numeric = int | float

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
            raise TypeError('data_array must be an instance of scipp.DataArray.')

        self._validate_data_array_coordinate(data_array, 'tof', 'time-of-flight information', 't', 'time', 's')

        if any(data_array.coords['tof'].to(unit='s') < sc.scalar(0, unit='s')):
            raise ValueError('time_of_flight values must be non-negative.')

        if 'x' in data_array.coords:
            self._validate_data_array_coordinate(data_array, 'x', 'pixels', 'x', 'length', 'm')
            self._pixel_positions_x = data_array.coords['x']
        elif 'x' not in data_array.dims:
            raise ValueError("data array must have an 'x' dimension.")

        if 'y' in data_array.coords:
            self._validate_data_array_coordinate(data_array, 'y', 'pixels', 'y', 'length', 'm')
            self._pixel_positions_y = data_array.coords['y']
        elif 'y' not in data_array.dims:
            raise ValueError("data array must have an 'y' dimension.")

        self._time_of_flight = data_array.coords['tof']

        super().__init__(unique_name=unique_name, display_name=display_name)

        self._full_data_array = data_array.copy(deep=False)

        # Ensure x and y coordinates are in edge format for consistent ROIs across rebinning
        if 'x' in self._full_data_array.coords and not self._full_data_array.coords.is_edges('x'):
            self._full_data_array.coords['x'] = Measurement._to_edges(self._full_data_array.coords['x'])
        if 'y' in self._full_data_array.coords and not self._full_data_array.coords.is_edges('y'):
            self._full_data_array.coords['y'] = Measurement._to_edges(self._full_data_array.coords['y'])

        # Fallback for when no x/y coordinates are provided
        # For tracking original pixels when rebinning, add pixel indices as coordinates
        self._full_data_array.coords['x_pixels'] = sc.arange('x', 0, self._full_data_array.sizes['x'] + 1, 1)
        self._full_data_array.coords['y_pixels'] = sc.arange('y', 0, self._full_data_array.sizes['y'] + 1, 1)

        self._regions_of_interest = []

    @classmethod
    def from_scitiff(
        cls, filename: str | Path, unique_name: str | None = None, display_name: str | None = None
    ) -> Measurement:  # noqa: E501
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
            raise TypeError('filename must be a string or Path object.')
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
        time_of_flights: sc.Variable | np.array,
        x_positions: sc.Variable | np.array | None = None,
        y_positions: sc.Variable | np.array | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ) -> Measurement:
        """
        Create a Measurement instance by loading data from a TIFF stack file.

        Parameters
        ----------
        filename : str | Path
            Path to the TIFF stack file.
        time_of_flights : sc.Variable | np.array
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
            raise TypeError('filename must be a string or Path object.')
        try:
            data_array = load_scitiff(filename)['image']
        except Exception as e:
            raise FileNotFoundError(f"Failed to load TIFF stack file '{filename}': {e}") from e

        try:
            data_array = data_array.rename_dims({'dim_0': 't', 'dim_1': 'y', 'dim_2': 'x'})
        except Exception as e:
            raise RuntimeError(f"Failed to rename dimensions for TIFF stack file '{filename}': {e}") from e

        time_of_flights = cls._validate_provided_coord(
            data_array, time_of_flights, 'time_of_flight', 't', 'frames in the TIFF stack', 'time', 's'
        )
        data_array.coords['tof'] = time_of_flights

        if x_positions is not None:
            x_positions = cls._validate_provided_coord(
                data_array, x_positions, 'x_positions', 'x', 'pixels in the x dimension', 'length', 'm'
            )
            data_array.coords['x'] = x_positions

        if y_positions is not None:
            y_positions = cls._validate_provided_coord(
                data_array, y_positions, 'y_positions', 'y', 'pixels in the y dimension', 'length', 'm'
            )
            data_array.coords['y'] = y_positions

        instance = cls(data_array=data_array, unique_name=unique_name, display_name=display_name)
        return instance

    def rebin(self, dimensions: dict[str, Numeric]) -> None:
        """
        Rebin the measurement image stack. This operation reduces the resolution of the data by combining adjacent pixels or time bins.
        The rebinned dimensions must be evenly divisible by their specific rebin factor.

        To revert to the original data, provide a dictionary with all rebin factors set to 1.

        Parameters
        ----------
        dimensions : dict[str, int]
            A dictionary specifying the rebinning factors for each dimension.
            For example, {'t': 2} will rebin the time dimension by a factor of 2.
        """  # noqa: E501
        if not isinstance(dimensions, dict):
            raise TypeError('dimensions must be a dictionary mapping dimension names to rebin factors.')
        if all(isinstance(value, Numeric) and value == 1 for value in dimensions.values()):  # Reverts to original data
            self.revert_rebin()
            return
        if 't' in dimensions:
            raise ValueError("Rebinning of the time-of-flight ('t') dimension is yet not supported.")
        for dim, value in dimensions.items():
            if not isinstance(dim, str):
                raise TypeError(f'Dimension keys must be strings. Got {type(dim)} for {dim} instead.')
            if dim not in self._full_data_array.dims:
                raise KeyError(
                    f"Dimension '{dim}' not a valid dimension for rebinning. Should be one of {self._full_data_array.dims}."
                )
            if (
                not (isinstance(value, int) or (isinstance(value, float) and value.is_integer())) and value < 1
            ):  # I allow eg. 2.0 as well as 2  # noqa: E501
                raise ValueError(f"Rebin size for dimension '{dim}' must be a positive integer of at least 1.")
            if self._full_data_array.sizes[dim] % value != 0:
                raise ValueError(
                    f"Dimension '{dim}' with size {self._full_data_array.sizes[dim]} is not evenly divisible by rebin size {value}."  # noqa: E501
                )  # noqa: E501
        self._rebinned_data_array = essimaging.tools.analysis.resize(self._full_data_array, sizes=dimensions, method='mean')
        # Update coordinates after rebinning. Can be removed when scipp supports resizing with coordinates.
        if 'x' in dimensions:
            self._rebinned_data_array.coords['x_pixels'] = self._full_data_array.coords['x_pixels'][:: dimensions['x']]  # Bin-edge
            if 'x' in self._full_data_array.coords:
                self._rebinned_data_array.coords['x'] = self._full_data_array.coords['x'][:: dimensions['x']]  # Bin-edge
        if 'y' in dimensions:
            self._rebinned_data_array.coords['y_pixels'] = self._full_data_array.coords['y_pixels'][:: dimensions['y']]  # Bin-edge
            if 'y' in self._full_data_array.coords:
                self._rebinned_data_array.coords['y'] = self._full_data_array.coords['y'][:: dimensions['y']]  # Bin-edge

    @property
    def _data_array(self) -> sc.DataArray:
        """
        Get the current data array, either rebinned or the original full resolution.
        """
        if self._rebinned_data_array is not None:
            return self._rebinned_data_array
        return self._full_data_array
    
    @_data_array.setter
    def _data_array(self, value: sc.DataArray) -> None:
        raise AttributeError("Cannot set _data_array, it is a read-only property.")

    def revert_rebin(self) -> None:
        """
        Revert any rebinning applied to the measurement data, restoring it to its original resolution.
        """
        if self._rebinned_data_array is not None:
            del self._rebinned_data_array

    def _validate_data_array_coordinate(
        self,
        data_array: sc.DataArray,
        coord_name: str,
        coord_context: str,
        expected_dim: str,
        expected_dim_string: str,
        expected_unit: str,
    ) -> None:
        if data_array.coords.get(coord_name) is None or data_array.coords[coord_name].shape == ():
            raise ValueError(f"data array must contain '{coord_name}' coordinate for {coord_context}.")
        if data_array.coords[coord_name].dim != expected_dim:
            raise ValueError(f"'{coord_name}' coordinate must be of dimension '{expected_dim}'.")
        try:
            data_array.coords[coord_name].to(unit=expected_unit)
        except UnitError:
            raise UnitError(
                f"'{coord_name}' coordinate must have a unit of {expected_dim_string}, such as ('{expected_unit}')."
            ) from None  # noqa: E501

    @staticmethod
    def _to_edges(centers: sc.Variable) -> sc.Variable:
        """
        Convenience method to convert center coordinates to edge coordinates.
        """
        interior_edges = sc.midpoints(centers)
        return sc.concat(
            [
                2 * centers[0] - interior_edges[0],
                interior_edges,
                2 * centers[-1] - interior_edges[-1],
            ],
            dim=centers.dim,
        )

    @staticmethod
    def _validate_provided_coord(
        data_array: sc.DataArray,
        coord: sc.Variable | np.array,
        coord_name: str,
        dim: str,
        length_context: str,
        expected_dim_string: str,
        expected_unit: str,
    ) -> sc.Variable:
        if not isinstance(coord, (sc.Variable, np.ndarray)):
            raise TypeError(f'{coord_name} must be a scipp Variable or a numpy Array.')
        if len(coord) not in (data_array.sizes[dim], data_array.sizes[dim] + 1):
            raise ValueError(f'Length of {coord_name} array does not match the number of {length_context}.')
        if isinstance(coord, np.ndarray):
            coord = sc.array(dims=[dim], values=coord, unit=expected_unit)
        try:
            coord.to(unit=expected_unit)
        except UnitError:
            raise UnitError(f"{coord_name} must have a unit of {expected_dim_string}, such as '{expected_unit}'.") from None
        return coord
