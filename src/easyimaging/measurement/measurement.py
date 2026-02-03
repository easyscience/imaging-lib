#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ess.imaging as essimaging
import matplotlib
import numpy as np
import plopp as pp
import scipp as sc
from easyscience.base_classes import NewBase
from scipp import DimensionError
from scipp import UnitError
from scitiff import load_scitiff

from easyimaging.measurement.regions import RectROI

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

        if 'tof' in data_array.coords:
            self._validate_data_array_coordinate(data_array, 'tof', 't', 'time', 's')
        else:
            raise ValueError("data array must have a 'tof' coordinate for time-of-flight information.")

        if any(data_array.coords['tof'].to(unit='s') < sc.scalar(0, unit='s')):
            raise ValueError('time_of_flight values must be non-negative.')

        if 'x' in data_array.coords and 'y' in data_array.coords:
            self._validate_data_array_coordinate(data_array, 'x', 'x', 'length', 'm')
            self._validate_data_array_coordinate(data_array, 'y', 'y', 'length', 'm')
            self._has_physical_coords = True
        elif 'x' not in data_array.dims or 'y' not in data_array.dims:
            raise DimensionError("data array must have both 'x' and 'y' dimensions.")
        elif 'x' not in data_array.coords and 'y' not in data_array.coords:
            self._has_physical_coords = False
        else:
            raise ValueError("data array must have both 'x' and 'y' coordinates or neither.")

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
        non_finite_mask = ~sc.isfinite(self._full_data_array.data)
        self._full_data_array.masks['non_finite'] = non_finite_mask

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
            # self.revert_rebin() # If we want secondary rebins to work on the original data
            return
        sizes = dimensions.copy()
        if 't' in dimensions:
            raise ValueError("Rebinning of the time-of-flight ('t') dimension is yet not supported.")
        for dim, value in dimensions.items():
            if not isinstance(dim, str):
                raise TypeError(f'Dimension keys must be strings. Got {type(dim)} for {dim} instead.')
            if dim not in self._full_data_array.dims:
                raise KeyError(
                    f"Dimension '{dim}' not a valid dimension for rebinning. Should be one of {self._full_data_array.dims}."
                )
            if isinstance(value, float) and value.is_integer():  # I allow eg. 2.0 as well as 2
                value = int(value)
                dimensions[dim] = value  # This line can be removed when scipp resize support resizing with coordinates
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"Rebin size for dimension '{dim}' must be a positive integer of at least 1.")
            if self._full_data_array.sizes[dim] % value != 0:
                raise ValueError(
                    f"Dimension '{dim}' with size {self._full_data_array.sizes[dim]} is not evenly divisible by rebin size {value}."  # noqa: E501
                )  # noqa: E501
            sizes[dim] = int(self._data_array.sizes[dim] // value)  # Convert to target size
        temp_array = essimaging.tools.analysis.resize(self._data_array, sizes=sizes, method='mean')
        # ------------------------------ To be removed when scipp supports resizing with coordinates. -------------------------
        if 'x' in dimensions:
            temp_array.coords['x_pixels'] = self._data_array.coords['x_pixels'][:: dimensions['x']]  # Bin-edge
            if 'x' in self._full_data_array.coords:
                temp_array.coords['x'] = self._data_array.coords['x'][:: dimensions['x']]  # Bin-edge
        if 'y' in dimensions:
            temp_array.coords['y_pixels'] = self._data_array.coords['y_pixels'][:: dimensions['y']]  # Bin-edge
            if 'y' in self._full_data_array.coords:
                temp_array.coords['y'] = self._data_array.coords['y'][:: dimensions['y']]  # Bin-edge
        # ---------------------------------------------------------------------------------------------------------------------
        non_finite_mask = ~sc.isfinite(temp_array.data)
        temp_array.masks['non_finite'] = non_finite_mask
        self._rebinned_data_array = temp_array

    @property
    def _data_array(self) -> sc.DataArray:
        """
        Get the current data array, either rebinned or the original full resolution.
        """
        if hasattr(self, '_rebinned_data_array'):
            return self._rebinned_data_array
        return self._full_data_array

    @_data_array.setter
    def _data_array(self, value: sc.DataArray) -> None:
        raise AttributeError('Cannot set _data_array, it is a read-only property.')

    def revert_rebin(self) -> None:
        """
        Revert any rebinning applied to the measurement data, restoring it to its original resolution.
        """
        if self._rebinned_data_array is not None:
            del self._rebinned_data_array

    def plot(self, time_of_flight: int | sc.Variable | None = None, **kwargs) -> None:
        """
        Plot the measurement image at a specific time-of-flight.
        If no time-of-flight is provided, the plot will sum over all time-of-flight values.

        This method uses the plopp library for plotting:
        https://scipp.github.io/plopp/plotting/image-plot.html

        Parameters
        ----------
        time_of_flight : int | sc.Variable | None
            The time-of-flight value to plot. If None, the time-of-flight axis is summed up.
        kwargs : dict
            Additional keyword arguments to pass to the plotting function.
            See https://scipp.github.io/plopp/generated/plopp.plot.html for options.
        """
        if time_of_flight is None:
            title_suffix = ' (summed over TOF)'
        elif isinstance(time_of_flight, int):
            title_suffix = f' at TOF index {time_of_flight}'
        elif isinstance(time_of_flight, sc.Variable):
            title_suffix = f' at TOF={time_of_flight}'
        else:
            title_suffix = ''

        plot_kwargs_defaults = {
            'title': self.display_name + title_suffix,
            'clabel': 'Transmission',
            'cmin': 0.0,
            'cmax': 3.0,
            'mask_color': 'red',
        }
        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        if time_of_flight is None:
            plot = self._data_array.mean('t').plot(**plot_kwargs_defaults)
        elif isinstance(time_of_flight, int):
            plot = self._data_array['t', time_of_flight].plot(**plot_kwargs_defaults)
        elif isinstance(time_of_flight, sc.Variable):
            try:
                plot = self._data_array['tof', time_of_flight].plot(**plot_kwargs_defaults)
            except UnitError:
                raise UnitError("time_of_flight variable must have a unit of time such as 's'") from None
        else:
            raise TypeError('time_of_flight must be an integer, scipp Variable, or None.')
        if self._is_notebook():
            return plot
        else:
            plot.show()

    def slider_plot(self, **kwargs) -> None:
        """
        Launch an interactive slider plot for exploring the measurement data.

        This method uses the plopp library for interactive slicing:
        https://scipp.github.io/plopp/plotting/slicer-plot.html

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to the slicer function.
            See https://scipp.github.io/plopp/generated/plopp.slicer.html for options.
        """
        slicer_kwargs_defaults = {
            'title': self.display_name + ' - Time of Flight Slicer',
            'clabel': 'Transmission',
            'cmin': 0.0,
            'cmax': 3.0,
            'mask_color': 'red',
            'coords': 'tof',
        }
        # Overwrite defaults with any user-provided kwargs
        slicer_kwargs_defaults.update(kwargs)

        if self._is_notebook():
            if matplotlib.get_backend() == 'widget':
                return pp.slicer(self._data_array, keep=['x', 'y'], **slicer_kwargs_defaults)
            else:
                raise RuntimeError(
                    'Interactive slicer requires the matplotlib "widget" backend in Jupyter notebooks. \n'
                    'To set it, run "%matplotlib widget" in a notebook cell before launching the slicer.'
                )
        else:
            raise RuntimeError('Interactive slicer is only supported in Jupyter notebooks.')

    def spectrum_inspector(self, **kwargs) -> None:
        """
        Launch an interactive spectrum inspector plot for exploring the measurement data.

        This method uses the plopp library for interactive inspection:
        https://scipp.github.io/plopp/plotting/inspector-plot.html

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments to pass to the inspector function.
            See https://scipp.github.io/plopp/generated/plopp.inspector.html for options.
        """
        inspector_kwargs_defaults = {
            'title': self.display_name + ' - Spectrum Inspector',
            'clabel': 'Transmission',
            'cmin': 0.0,
            'cmax': 3.0,
            'mask_color': 'red',
            'ymax': 3.0,
            'ymin': 0.0,
        }
        # Overwrite defaults with any user-provided kwargs
        inspector_kwargs_defaults.update(kwargs)

        if self._is_notebook():
            if matplotlib.get_backend() == 'widget':
                return pp.inspector(self._data_array, dim='t', orientation='vertical', **inspector_kwargs_defaults)
            else:
                raise RuntimeError(
                    'Interactive spectrum inspector requires the matplotlib "widget" backend in Jupyter notebooks. \n'  # noqa: E501
                    'To set it, run "%matplotlib widget" in a notebook cell before launching the inspector.'
                )
        else:
            raise RuntimeError('Interactive spectrum inspector is only supported in Jupyter notebooks.')

    def spectrum(self, roi: RectROI = None) -> sc.DataArray:
        """
        Extract the spectrum (intensity vs. time-of-flight) for a specified region of interest (ROI).
        If no ROI is provided, the spectrum is calculated over the entire image.

        Parameters
        ----------
        roi : RectROI
            The region of interest for which to extract the spectrum.

        Returns
        -------
        sc.DataArray
            A DataArray containing the spectrum data.
        """
        if roi is not None and not isinstance(roi, RectROI):
            raise TypeError('roi must be an instance of RectROI or None.')

        if roi is None:
            spectrum_data = self._data_array.mean(dim=['x', 'y'])
        elif 'x' in self._data_array.coords and 'y' in self._data_array.coords:
            x_slice, y_slice = roi.slice()
        else:
            x_slice, y_slice = roi.pixel_slice()
        return self._data_array['x', x_slice]['y', y_slice].mean(dim=['x', 'y'])

    def _is_notebook(self) -> bool:
        """
        Check if the code is running in a Jupyter notebook environment.
        """
        try:
            shell = get_ipython().__class__.__name__  # pyright: ignore[reportUndefinedVariable]
            if shell == 'ZMQInteractiveShell':
                return True  # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (possibly other IDE)
        except NameError:
            return False  # Probably standard Python interpreter

    def _validate_data_array_coordinate(
        self,
        data_array: sc.DataArray,
        coord_name: str,
        expected_dim: str,
        expected_dim_string: str,
        expected_unit: str,
    ) -> None:
        if data_array.coords[coord_name].dims != (expected_dim,):
            raise DimensionError(f"'{coord_name}' coordinate must be of dimension '{expected_dim}'.")
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
