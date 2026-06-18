# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path

import ess.imaging as essimaging
import numpy as np
import plopp as pp
import scipp as sc
from easyscience.base_classes import EasyList
from easyscience.base_classes import NewBase
from scipp import DimensionError
from scipp import UnitError
from scitiff import load_scitiff
from scitiff import save_scitiff
from scitiff.io import ImageJMetadataNotFoundWarning

from ..regions_of_interest import RectangleROI
from ..utils import _is_notebook
from ..utils import _to_edges

Numeric = int | float


class Measurement(NewBase):
    """Object responsible for managing and inspecting the **normalized** measurement data of a time-of-flight neutron imaging
     experiment. This object can be created in 3 ways:

    - by loading the data from a [SciTiff](https://scipp.github.io/scitiff/) file with the
     [from_scitiff][.from_scitiff] method.<br>
     This is the recommended approach, as the [SciTiff](https://scipp.github.io/scitiff/) format natively contains all the
     necessary metadata for the analysis of the measurement data.
    - by directly providing a properly formatted [`sc.DataArray`][scipp.DataArray] to the constructor.<br>
     This is mostly useful for when working directly in a reduction notebook.
    - by loading from a regular .tiff stack file with the [from_tiff_stack][.from_tiff_stack] method.<br>
     This method is provided to support older datasets that are not in the SciTiff format.

    Examples of all 3 methods are supplied below.

    When the [Measurement][.] object is created, a mask is automatically applied to filter out non-finite values in the data.

    Example
    -------
    **Creating a [Measurement][.] instance by loading from a [SciTiff](https://scipp.github.io/scitiff/) file**

    Using the example data provided by the library:
    ```python
    from easyimaging.datasets import iron_alpha_scitiff
    from easyimaging import Measurement

    scitiff_path = iron_alpha_scitiff()

    measurement = Measurement.from_scitiff(filename=scitiff_path)
    ```

    **Creating a [Measurement][.] instance by directly providing a [`sc.DataArray`][scipp.DataArray]**

    Note that the dimension names must be exactly `'x'`, `'y'`, and `'t'`, and that the coordinate names must be exactly
     `'x'`, `'y'`, and `'tof'` to be recognised by the constructor, and that the `'tof'` coordinate must be provided, whereas
     `'x'` and `'y'` are optional:
    ```python
    import scipp as sc
    from easyimaging import Measurement

    tof = sc.arange('t', 0, 10, 1, unit='s')
    x = sc.arange('x', 0, 7, 1, unit='m')  # Optional
    y = sc.arange('y', 0, 7, 1, unit='m')  # Optional
    coords = {'tof': tof, 'x': x, 'y': y}
    data = sc.ones(dims=['t', 'y', 'x'], shape=[10, 6, 6])

    data_array = sc.DataArray(data=data, coords=coords)

    measurement = Measurement(data_array=data_array)
    ```

    **Creating a [Measurement][.] instance by loading from a regular .tiff stack file**

    Using the example data provided by the library, note that the time_of_flights must be provided, whereas the x_positions
     and y_positions are optional:
    ```python
    from easyimaging.datasets import iron_alpha_tiff
    from easyimaging import Measurement
    import scipp as sc

    tiff_path = iron_alpha_tiff()

    measurement = Measurement.from_tiff_stack(
        filename=tiff_path,
        time_of_flights=sc.arange('t', 0, 1501, 1, unit='s'),  # mandatory
        # x_positions=sc.arange('x', 0, 50, 1, unit='mm'), # optional
        # y_positions=sc.arange('y', 0, 50, 1, unit='mm'), # optional
    )
    ```
    """

    def __init__(
        self,
        data_array: sc.DataArray,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """Initialize a Measurement instance.

        Parameters
        ----------
        data_array : sc.DataArray
            The measurement data in a [`sc.DataArray`][scipp.DataArray] with dimensions ``('x', 'y', 't')``.<br>
            Must have a ``'tof'`` coordinate for time-of-flight values in the dimension ``'t'``.<br>
            Optionally may include ``'x'`` and ``'y'`` coordinates for physical pixel positions.<br>
            Other coordinates in the [`sc.DataArray`][scipp.DataArray] are ignored.
        unique_name : str | None
            A unique identifier for the [`Measurement`][..]. Defaults to ``'Measurement'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Measurement`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Raises
        ------
        TypeError
            If ``data_array`` is not a [`sc.DataArray`][scipp.DataArray].
        ValueError
            If the ``data_array`` is missing the ``'tof'`` coordinate.<br>
            If the ``'tof'`` coordinate contains negative time-of-flight values.<br>
            If the ``data_array`` has only one of the ``'x'`` or ``'y'`` coordinates, but not both.<br>
        DimensionError
            If the ``'tof'`` coordinate does not have the dimension ``'t'``.<br>
            If the ``'x'`` and ``'y'`` coordinates do not have the dimensions ``'x'`` and ``'y'``, respectively, when
             provided.<br>
            If the ``data_array`` does not have both ``'x'`` and ``'y'`` dimensions.
        UnitError
            If the ``'tof'`` coordinate does not have a unit of time.<br>
            If the ``'x'`` and ``'y'`` coordinates do not have a unit of length, when provided.
        """
        if not isinstance(data_array, sc.DataArray):
            raise TypeError('data_array must be an instance of scipp.DataArray.')

        if 'tof' in data_array.coords:
            self._validate_data_array_coordinate(data_array, 'tof', 't', 'time', 's')
        else:
            raise ValueError("data array must have a 'tof' coordinate for time-of-flight information.")

        if any(data_array.coords['tof'].to(unit='s') < sc.scalar(0, unit='s')):
            raise ValueError('time_of_flight values must be non-negative.')

        # Coordinate names were changed in scitiff.
        if 'x_pixel_offset' in data_array.coords:
            data_array.coords['x'] = data_array.coords.pop('x_pixel_offset')
        if 'y_pixel_offset' in data_array.coords:
            data_array.coords['y'] = data_array.coords.pop('y_pixel_offset')

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
        if self._has_physical_coords and not self._full_data_array.coords.is_edges('x'):
            self._full_data_array.coords['x'] = _to_edges(self._full_data_array.coords['x'])
        if self._has_physical_coords and not self._full_data_array.coords.is_edges('y'):
            self._full_data_array.coords['y'] = _to_edges(self._full_data_array.coords['y'])

        # Fallback for when no x/y coordinates are provided
        # For tracking original pixels when rebinning, add pixel indices as coordinates
        self._full_data_array.coords['x_pixels'] = sc.arange('x', 0, self._full_data_array.sizes['x'] + 1, 1)
        self._full_data_array.coords['y_pixels'] = sc.arange('y', 0, self._full_data_array.sizes['y'] + 1, 1)

        self._regions_of_interest = EasyList(protected_types=(RectangleROI,))

        non_finite_mask = ~sc.isfinite(self._full_data_array.data)
        self._full_data_array.masks['non_finite'] = non_finite_mask

    @classmethod
    def from_scitiff(
        cls, filename: str | Path, unique_name: str | None = None, display_name: str | None = None
    ) -> Measurement:  # noqa: E501
        """Create a `Measurement` object by loading normalized image data from a **SciTiff** file.

        Time-of-flight values and other relevant metadata, such as the flight-path, are automatically extracted from the file,
        if present.

        Uses the [SciTiff](https://scipp.github.io/scitiff/) library's
         [`load_scitiff`][scitiff.load_scitiff] method internally to load the data.

        Example
        ----------
        An example of how to use this method is given in the [Measurement][..] class description.

        Parameters
        ----------
        filename : str | Path
            The path to the [SciTiff](https://scipp.github.io/scitiff/) file.
        unique_name : str | None
            A unique identifier for the [`Measurement`][..]. Defaults to ``'Measurement'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Measurement`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Returns
        -------
        Measurement
            An instance of the [`Measurement`][..] class containing the loaded data.

        Raises
        ------
        TypeError
            If ``filename`` is not a string or a [`Path`](https://docs.python.org/3/library/pathlib.html) object.
        RuntimeError
            If the ``filename`` cannot be loaded or is not a valid [SciTiff](https://scipp.github.io/scitiff/) file.
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
        time_of_flights: sc.Variable | np.ndarray,
        x_positions: sc.Variable | np.ndarray | None = None,
        y_positions: sc.Variable | np.ndarray | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ) -> Measurement:
        """Create a `Measurement` object by loading normalized image data from a regular **TIFF** stack file.

        Since a regular TIFF file does not contain the necessary metadata for Bragg-edge imaging,
         such as **time-of-flight** values for each frame in the image stack, they have to be provided manually.

        The optional metadata can also be set after the object has already been created.

        Example
        ----------
        An example of how to use this method is given in the [Measurement][..] class description.

        Parameters
        ----------
        filename : str | Path
            The path to the TIFF stack file.
        time_of_flights : sc.Variable | np.ndarray
            A 1-dimensional array of <nobr>time-of-flight</nobr> values corresponding to the frames in the TIFF stack.<br>
            If a [numpy array][numpy.ndarray] is provided, it is assumed to be in units of seconds.
        x_positions : sc.Variable | np.ndarray | None
            A 1-dimensional array of physical x-axis positions for the TIFF stack pixels.<br>
            If a [numpy array][numpy.ndarray] is provided, it is assumed to be in units of meters.
        y_positions : sc.Variable | np.ndarray | None
            A 1-dimensional array of physical y-axis positions for the TIFF stack pixels.<br>
            If a [numpy array][numpy.ndarray] is provided, it is assumed to be in units of meters.
        unique_name : str | None
            A unique identifier for the [`Measurement`][..].<br>
            Defaults to ``'Measurement'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Measurement`][..]. Defaults to [`unique_name`][..unique_name] if not provided.

        Returns
        -------
        Measurement
            An instance of the [`Measurement`][..] class containing the loaded and provided data.

        Raises
        ------
        TypeError
            If ``filename`` is not a string or a [`Path`](https://docs.python.org/3/library/pathlib.html) object.<br>
            If ``time_of_flights``, ``x_positions``, or ``y_positions`` are not a [`sc.Variable`][scipp.Variable] or a
             [numpy.ndarray][numpy.ndarray].<br>
        ValueError
            If ``time_of_flights`` is not 1-dimensional with the same length as the number of frames in the TIFF stack.<br>
            If either ``x_positions`` or ``y_positions`` are not 1-dimensional with the same length as the number of pixels in
             the corresponding dimension of the TIFF stack.<br>
            If the ``'time_of_flights'`` contain negative values.<br>
        UnitError
            If ``time_of_flights`` is a [`sc.Variable`][scipp.Variable] that does not have a unit of time.<br>
            If either ``x_positions`` or ``y_positions`` are a [`sc.Variable`][scipp.Variable] without a unit of
             length.
        RuntimeError
            If the ``filename`` cannot be loaded.<br>
            If the file is a [SciTiff](https://scipp.github.io/scitiff/) file: Use [from_scitiff][..from_scitiff] instead.
        """
        if not isinstance(filename, (str, Path)):
            raise TypeError('filename must be a string or Path object.')

        # We know this is not a scitiff, so supress that warning
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ImageJMetadataNotFoundWarning)
            try:
                data_array = load_scitiff(filename)['image']
            except Exception as e:
                raise RuntimeError(f"Failed to load TIFF stack file '{filename}': {e}") from e

        try:
            data_array = data_array.rename_dims({'dim_0': 't', 'dim_1': 'y', 'dim_2': 'x'})
        except Exception as e:
            raise RuntimeError(f"Failed to rename dimensions for TIFF stack file '{filename}': {e}") from e

        time_of_flights = cls._validate_provided_coord(
            data_array=data_array,
            coord=time_of_flights,
            coord_name='time_of_flight',
            dim='t',
            length_context='frames in the TIFF stack',
            expected_dim_string='time',
            expected_unit='s',
        )
        data_array.coords['tof'] = time_of_flights

        if x_positions is not None:
            x_positions = cls._validate_provided_coord(
                data_array=data_array,
                coord=x_positions,
                coord_name='x_positions',
                dim='x',
                length_context='pixels in the x dimension',
                expected_dim_string='length',
                expected_unit='m',
            )
            data_array.coords['x'] = x_positions

        if y_positions is not None:
            y_positions = cls._validate_provided_coord(
                data_array=data_array,
                coord=y_positions,
                coord_name='y_positions',
                dim='y',
                length_context='pixels in the y dimension',
                expected_dim_string='length',
                expected_unit='m',
            )
            data_array.coords['y'] = y_positions

        instance = cls(data_array=data_array, unique_name=unique_name, display_name=display_name)
        return instance

    def save_scitiff(self, filename: str | Path) -> None:
        """Save the `Measurement` data to a **SciTiff** file with the time-of-flight values and other provided metadata.

        Note that [SciTiff](https://scipp.github.io/scitiff/) does not support ``float64`` precision numbers, so if the
         [`Measurement`][..] contains data of type ``float64``, it will be downcast to ``float32`` when saving, causing a small
          loss of precision.

        Parameters
        ----------
        filename : str | Path
            Path to the output [SciTiff](https://scipp.github.io/scitiff/) file.

        Raises
        ------
        TypeError
            If ``filename`` is not a string or a [`Path`](https://docs.python.org/3/library/pathlib.html) object.
        RuntimeError
            If the file cannot be written to the specified ``filename`` path.

        Warns
        -----
        UserWarning
            If the [Measurement][..] object's data is of type ``float64``; It will be downcast to ``float32``.
        """
        if not isinstance(filename, (str, Path)):
            raise TypeError('filename must be a string or Path object.')
        try:
            nan_mask = self._data_array.masks.pop('non_finite')
            if str(self._data_array.dtype) == 'float64':
                warnings.warn(
                    'The data array is of type float64, which is not directly supported by the SciTIFF format. '
                    'It will be downcast to float32 when saving, which may result in loss of precision. '
                )
                save_scitiff(self._data_array.astype('float32'), filename)
            else:
                save_scitiff(self._data_array, filename)
        except Exception as e:
            raise RuntimeError(f"Failed to save SciTIFF file '{filename}': {e}") from e
        finally:
            self._data_array.masks['non_finite'] = nan_mask  # Ensure mask is restored

    @property
    def data_array_copy(self) -> sc.DataArray:
        """Get a full deep copy of the current data array. This attribute is **read-only**.

        Returns the rebinned data array if rebinning has been applied, otherwise the
        original full-resolution data array. Use with caution for large datasets to avoid memory issues.

        Returns
        -------
        sc.DataArray
            A deep copy of the current data array.
        """
        return self._data_array.copy(deep=True)

    # Can't have setters due to MkDocstrings then showing the property as writable.
    # @data_array_copy.setter
    # def data_array_copy(self, value: sc.DataArray) -> None:
    #     """Raise AttributeError — ``data_array_copy`` is a read-only property.

    #     Raises
    #     ------
    #     AttributeError
    #         Always. To use a different data array, create a new :class:`Measurement` instance.
    #     """
    #     raise AttributeError(
    #         'Cannot set data_array, it is a read-only property. '
    #         'Please make a new Measurement instance if you want to use a different data array.'
    #     )

    @property
    def _data_array(self) -> sc.DataArray:
        """Get the current data array, either rebinned or the original full resolution.

        Returns
        -------
        sc.DataArray
            The rebinned data array if rebinning has been applied, otherwise the
            original full-resolution data array.
        """
        if hasattr(self, '_rebinned_data_array'):
            return self._rebinned_data_array
        return self._full_data_array

    @_data_array.setter
    def _data_array(self, value: sc.DataArray) -> None:
        """Raise AttributeError — ``_data_array`` is a read-only property.

        Raises
        ------
        AttributeError
            Always.
        """
        raise AttributeError('Cannot set _data_array, it is a read-only property.')

    @property
    def x_positions(self) -> sc.Variable:
        """The x-coordinates of the physical positions of the pixels.

        Parameters
        ----------
        value : sc.Variable | np.ndarray
            The new x-coordinate positions to set.<br>
            If a numpy array is provided, the unit is assumed to be meters.

        Returns
        -------
        sc.Variable
            A copy of the physical x-coordinate positions as a [`sc.Variable`][scipp.Variable].

        Raises
        ------
        ValueError
            If physical coordinate positions are not set (use
            [`set_physical_coord_positions`][..set_physical_coord_positions] to set)<br>
            If ``value`` is not 1-dimensional with the same length as the number of pixels (or pixel-edges) in the x-dimension.
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable] or [`np.ndarray`][numpy.ndarray].
        UnitError
            If ``value`` is a [`sc.Variable`][scipp.Variable] and does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._data_array.coords['x'].copy()
        else:
            raise ValueError('Physical coordinate positions are not set for this Measurement.')

    @x_positions.setter
    def x_positions(self, value: sc.Variable | np.ndarray) -> None:
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if not self._has_physical_coords:
            raise ValueError(
                'Cannot set x_positions before setting all physical coordinate positions. '
                'Please use the set_physical_coord_positions method.'
            )
        value = self._validate_provided_coord(
            data_array=self._data_array,
            coord=value,
            coord_name='x_positions',
            dim='x',
            length_context='pixels in the x dimension',
            expected_dim_string='length',
            expected_unit='m',
        )
        self._data_array.coords['x'] = value

    @property
    def y_positions(self) -> sc.Variable:
        """The y-coordinates of the physical positions of the pixels.

        Parameters
        ----------
        value : sc.Variable | np.ndarray
            The new y-coordinate positions to set.<br>
            If a numpy array is provided, the unit is assumed to be meters.

        Returns
        -------
        sc.Variable
            A copy of the physical y-coordinate positions as a [`sc.Variable`][scipp.Variable].

        Raises
        ------
        ValueError
            If physical coordinate positions are not set (use
            [`set_physical_coord_positions`][..set_physical_coord_positions] to set)<br>
            If ``value`` is not 1-dimensional with the same length as the number of pixels (or pixel-edges) in the y-dimension.
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable] or [`np.ndarray`][numpy.ndarray].
        UnitError
            If ``value`` is a [`sc.Variable`][scipp.Variable] and does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._data_array.coords['y'].copy()
        else:
            raise ValueError('Physical coordinate positions are not set for this Measurement.')

    @y_positions.setter
    def y_positions(self, value: sc.Variable | np.ndarray) -> None:
        # Setters have no docstrings. They should be written in the getter docstring instead.
        if not self._has_physical_coords:
            raise ValueError(
                'Cannot set y_positions before setting all physical coordinate positions. '
                'Please use the set_physical_coord_positions method.'
            )
        value = self._validate_provided_coord(
            data_array=self._data_array,
            coord=value,
            coord_name='y_positions',
            dim='y',
            length_context='pixels in the y dimension',
            expected_dim_string='length',
            expected_unit='m',
        )
        self._data_array.coords['y'] = value

    def set_physical_coord_positions(
        self, x_positions: sc.Variable | np.ndarray, y_positions: sc.Variable | np.ndarray
    ) -> None:
        """Set the physical xy-coordinate positions for the pixels of the measurement image stack.

        Parameters
        ----------
        x_positions : sc.Variable | np.ndarray
            A 1-dimensional array of physical x-axis positions for the image stack pixels.<br>
            If a [numpy array][numpy.ndarray] is provided, it is assumed to be in units of meters.
        y_positions : sc.Variable | np.ndarray
            A 1-dimensional array of physical y-axis positions for the image stack pixels.<br>
            If a [numpy array][numpy.ndarray] is provided, it is assumed to be in units of meters.

        Raises
        ------
        TypeError
            If either of ``x_positions`` or ``y_positions`` are not a [`sc.Variable`][scipp.Variable] or a
             [numpy.ndarray][numpy.ndarray].
        ValueError
            If either ``x_positions`` or ``y_positions`` are not 1-dimensional with the same length as the number of pixels in
             the corresponding dimension of the image stack.
        UnitError
            If either ``x_positions`` or ``y_positions`` are a [`sc.Variable`][scipp.Variable] without a unit of
             length.
        """
        x_positions = self._validate_provided_coord(
            data_array=self._data_array,
            coord=x_positions,
            coord_name='x_positions',
            dim='x',
            length_context='pixels in the x dimension',
            expected_dim_string='length',
            expected_unit='m',
        )
        y_positions = self._validate_provided_coord(
            data_array=self._data_array,
            coord=y_positions,
            coord_name='y_positions',
            dim='y',
            length_context='pixels in the y dimension',
            expected_dim_string='length',
            expected_unit='m',
        )
        self._data_array.coords['x'] = x_positions
        self._data_array.coords['y'] = y_positions
        self._has_physical_coords = True

    def delete_physical_coord_positions(self) -> None:
        """Delete the physical coordinate positions for the pixels of the measurement image stack.

        Raises
        ------
        ValueError
            If the [Measurement][..] does not currently have physical coordinate positions set.
        """
        if self._has_physical_coords:
            del self._data_array.coords['x']
            del self._data_array.coords['y']
            self._has_physical_coords = False
        else:
            raise ValueError('Cannot delete physical coordinate positions because they are not set.')

    @property
    def time_of_flights(self) -> sc.Variable:
        """The time-of-flight values of the measurement.

        Parameters
        ----------
        value : sc.Variable | np.ndarray
            The new time-of-flight values to set.<br>
            If a numpy array is provided, the unit is assumed to be seconds.

        Returns
        -------
        sc.Variable
            A copy of the time-of-flight values as a [`sc.Variable`][scipp.Variable].

        Raises
        ------
        ValueError
            If ``value`` is not 1-dimensional with the same length as the number of frames.<br>
            If any element in ``value`` is negative.
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable] or [`np.ndarray`][numpy.ndarray].
        UnitError
            If ``value`` is a [`sc.Variable`][scipp.Variable] and does not have a unit of time.
        """
        return self._data_array.coords['tof'].copy()

    @time_of_flights.setter
    def time_of_flights(self, value: sc.Variable | np.ndarray) -> None:
        # Setters have no docstrings. They should be written in the getter docstring instead.
        value = self._validate_provided_coord(
            data_array=self._data_array,
            coord=value,
            coord_name='time_of_flights',
            dim='t',
            length_context='frames in the measurement',
            expected_dim_string='time',
            expected_unit='s',
        )
        if any(value.to(unit='s') < sc.scalar(0, unit='s')):
            raise ValueError('time_of_flight values must be non-negative.')
        self._data_array.coords['tof'] = value

    @property
    def regions_of_interest(self) -> EasyList[RectangleROI]:
        """The list of regions of interest (ROIs) defined for this measurement.<br> This attribute is **read-only**.

        The returned list is an [EasyList][easyscience.base_classes.EasyList] of
         [regions of interest][...regions_of_interest.RectangleROI] objects.
        To modify the list or set new ROIs, interact directly with the returned list.

        Example
        -------
        ```python
        measurement.regions_of_interest.append(new_roi)  # Add a new ROI to the measurement
        measurement.regions_of_interest.remove('existing_roi')  # Remove an existing ROI
        ```

        Returns
        -------
        EasyList[RectangleROI]
            A list of the regions of interest (ROIs) attached to this measurement.
        """
        return self._regions_of_interest

    # Can't have setters due to MkDocstrings then showing the property as writable.
    # @regions_of_interest.setter
    # def regions_of_interest(self, value: EasyList[RectangleROI]) -> None:
    #     """Raise AttributeError — ``regions_of_interest`` is a read-only property.

    #     Raises
    #     ------
    #     AttributeError
    #         Always. Add or remove ROIs directly from the list instead.
    #     """
    #     raise AttributeError(
    #         'Cannot set regions_of_interest, it is a read-only property. '
    #         'Please simply add or remove ROIs directly from the list.'
    #     )

    def rebin(self, dimensions: dict[str, Numeric]) -> None:
        """Rebin the measurement image stack.

        This operation increases the signal-to-noise ratio of the [Measurement][..] data, at the cost of a reduced resolution,
         by averaging adjacent pixels.<br>
        The rebinned dimensions must be evenly divisible by their specific rebin factors, as the operation is otherwise
         ill-defined.

        Repeating this operation will rebin the already rebinned data, not the original data.<br>
        This operation can be undone by using the [revert_rebin][..revert_rebin] method,
         which restores the original full-resolution data.

        Example
        -------
        Setting up a [Measurement][..]:
        ```python
        import scipp as sc
        from easyimaging import Measurement

        tof = sc.arange('t', 0, 10, 1, unit='s')
        coords = {'tof': tof}
        data = sc.ones(dims=['t', 'y', 'x'], shape=[10, 6, 6])

        data_array = sc.DataArray(data=data, coords=coords)

        measurement = Measurement(data_array=data_array)
        ```
        Halve the spatial resolution by rebinning by a factor of 2 in both dimensions:
        ```python
        measurement.rebin({'x': 2, 'y': 2})
        ```
        You can also rebin by different factors in the different dimensions,
         if the `Measurement` data dimensions are evenly divisible by those factors:
        ```python
        measurement.rebin({'x': 3, 'y': 2})
        ```
        You can also rebin only one dimension, leaving the other unchanged:
        ```python
        measurement.rebin({'x': 2})
        ```

        Parameters
        ----------
        dimensions : dict[str, int | float]
            A dictionary of dimension-rebin factor pairs, specifying the rebinning factors for each dimension.

        Raises
        ------
        TypeError
            If ``dimensions`` is not a `dict`.<br>
            If any keys in ``dimensions`` are not strings.<br>
            If a rebin factor is not a positive integer (or integer-valued float).
        ValueError
            If a dimension size is not evenly divisible by the requested rebin factor.
        KeyError
            If rebinning of the ``'t'`` dimension is requested (not yet supported).<br>
            If rebinning is attempted on a non-existing dimension.
        """
        if not isinstance(dimensions, dict):
            raise TypeError('dimensions must be a dictionary mapping dimension names to rebin factors.')
        if all(isinstance(value, Numeric) and value == 1 for value in dimensions.values()):  # Reverts to original data
            # self.revert_rebin() # If we want secondary rebins to work on the original data
            return
        sizes = dimensions.copy()
        if 't' in dimensions:
            raise KeyError("Rebinning of the time-of-flight ('t') dimension is not yet supported.")
        for dim, value in dimensions.items():
            if not isinstance(dim, str):
                raise TypeError(f'Dimension keys must be strings. Got {type(dim)} for {dim} instead.')
            if dim not in self._full_data_array.dims:
                raise KeyError(
                    f"Dimension '{dim}' not a valid dimension for rebinning. Should be one of {self._full_data_array.dims}."
                )
            if isinstance(value, float) and value.is_integer():  # I allow eg. 2.0 as well as 2
                value = int(value)
            if not isinstance(value, int) or value < 1:
                raise TypeError(f"Rebin size for dimension '{dim}' must be a positive integer of at least 1.")
            if self._full_data_array.sizes[dim] % value != 0:
                raise ValueError(
                    f"Dimension '{dim}' with size {self._full_data_array.sizes[dim]} is not"
                    f' evenly divisible by the requested rebin factor {value}.'
                )
            sizes[dim] = int(self._data_array.sizes[dim] // value)  # Convert to target size
        temp_array = essimaging.tools.analysis.resize(self._data_array, sizes=sizes, method='mean')
        non_finite_mask = ~sc.isfinite(temp_array.data)
        temp_array.masks['non_finite'] = non_finite_mask
        self._rebinned_data_array = temp_array

    def revert_rebin(self) -> None:
        """Revert any rebinning applied to the measurement data, restoring it to its original resolution.

        This method undos the rebinning performed by the [rebin][..rebin] method, if it has been applied.

        Warns
        ------
        UserWarning
            If no rebinning has been applied and the measurement data is already in its original state.
        """
        if hasattr(self, '_rebinned_data_array'):
            del self._rebinned_data_array
        else:
            warnings.warn(
                'No rebinning to revert. The data array is already in its original state.',
                UserWarning,
            )

    # -------------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------- Plotting methods --------------------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------------------

    def plot(self, time_of_flight: int | sc.Variable | None = None, **kwargs) -> None:
        """Plot the 2d spatial measurement image data.

        If no `time-of-flight` is provided, the plot will average over all time-of-flight values.<br>
        If `time-of-flight` is provided as a [Scipp](https://scipp.github.io/) variable scalar, the nearest time-of-flight
         frame in time will be plotted.

        This method uses the [plopp](https://scipp.github.io/plopp/plotting/image-plot.html) library for plotting.

        To customize the plot appearance, additional keyword arguments can be passed to the underlying plopp plotting function.
        See [plopp.plot][] for which keyword arguments are available and how to use them.

        Parameters
        ----------
        time_of_flight : int | sc.Variable | None
            The time-of-flight value to plot, as an index (int) or a value (sc.Variable).<br>
            If None, the time-of-flight axis is averaged.
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function.<br>
            See [plopp.plot][] for options.

        Raises
        ------
        TypeError
            If ``time_of_flight`` is not an integer, a `sc.Variable` scalar, or ``None``.
        UnitError
            If a `sc.Variable` ``time_of_flight`` is provided without a unit of time.
        """
        if time_of_flight is None:
            title_suffix = ' (averaged over TOF)'
        elif isinstance(time_of_flight, int):
            title_suffix = f' at TOF index {time_of_flight}'
        elif isinstance(time_of_flight, sc.Variable) and not time_of_flight.sizes:
            try:
                time_of_flight.to(unit='s')
            except UnitError:
                raise UnitError("time_of_flight variable must have a unit of time such as 's'") from None
            title_suffix = f' at TOF={time_of_flight.value} {time_of_flight.unit}'
        else:
            raise TypeError('time_of_flight must be an integer, scipp scalar, or None.')

        if time_of_flight is None:
            plot_array = self._data_array.mean('t')
        elif isinstance(time_of_flight, int):
            plot_array = self._data_array['t', time_of_flight]
        elif isinstance(time_of_flight, sc.Variable):
            plot_array = self._data_array['tof', time_of_flight]

        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults = self._plot_defaults()
        plot_kwargs_defaults['title'] = self.display_name + title_suffix
        plot_kwargs_defaults['cmax'] = min(3.0, float(plot_array.max().value * 1.1))
        plot_kwargs_defaults.update(kwargs)

        plot = plot_array.plot(**plot_kwargs_defaults)
        if _is_notebook():
            return plot
        else:
            plot.show()

    def slicer_plot(self, **kwargs) -> None:
        """Launch an interactive slicer plot for exploring the 2d spatial measurement image data.

        This method uses the [plopp](https://scipp.github.io/plopp/plotting/slicer-plot.html) library for interactive slicing.

        To customize the plot appearance, additional keyword arguments can be passed to the underlying plopp plotting function.
        See [plopp.slicer][] for which keyword arguments are available and how to use them.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the slicer function.<br>
            See [plopp.slicer][] for options.

        Raises
        ------
        RuntimeError
            If called outside a Jupyter notebook environment.
        """
        slicer_kwargs_defaults = self._plot_defaults()
        slicer_kwargs_defaults.update({
            'title': self.display_name + ' - Time of Flight Slicer',
            'keep': ['x_pixels', 'y_pixels'] if not self._has_physical_coords else ['x', 'y'],
            'mode': 'single',
        })
        slicer_kwargs_defaults['coords'].append('tof')

        # Overwrite defaults with any user-provided kwargs
        slicer_kwargs_defaults.update(kwargs)

        if _is_notebook():
            return pp.slicer(self._data_array, **slicer_kwargs_defaults)
        else:
            raise RuntimeError('Interactive slicer is only supported in Jupyter notebooks.')

    def spectrum_inspector(self, **kwargs) -> None:
        """Launch an interactive spectrum inspector plot for exploring the measurements time-of-flight spectrums.

        This method uses the [plopp](https://scipp.github.io/plopp/plotting/inspector-plot.html) library for interactive
         inspection.

        To customize the plot appearance, additional keyword arguments can be passed to the underlying plopp plotting function.
        See [plopp.inspector][] for which keyword arguments are available and how to use them.

        Interactive controls
        --------
        Click the ![](../../assets/images/crosshairs.png){width=15px} button in the toolbar on the left to activate/deactivate
         the spectrum investigator tool. When active, the following controls are available:

        - Left-click on a pixel to make a new point and view the spectrum for that pixel
        - Left-click and drag existing points to adjust which pixel's spectrum is being viewed
        - Middle-click on a point to delete it and remove its spectrum

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the inspector function.<br>
            See [plopp.inspector][] for options.

        Raises
        ------
        RuntimeError
            If called outside a Jupyter notebook environment.
        """
        inspector_kwargs_defaults = self._plot_defaults()
        inspector_kwargs_defaults['ymax'] = inspector_kwargs_defaults['cmax']
        inspector_kwargs_defaults.update({
            'title': self.display_name + ' - Spectrum Inspector',
            'ymin': 0.0,
            'dim': 'Time of flight',  # Due to https://github.com/scipp/plopp/issues/566
            'orientation': 'vertical',
            'operation': 'mean',
            'mode': 'point',
        })
        # Overwrite defaults with any user-provided kwargs
        inspector_kwargs_defaults.update(kwargs)

        # Hack to cirmunvent the bug in https://github.com/scipp/plopp/issues/566
        temp_array = self._data_array.drop_coords(self._data_array.coords)
        if self._has_physical_coords:
            temp_array.coords['x'] = self._data_array.coords['x']
            temp_array.coords['y'] = self._data_array.coords['y']
        temp_array.coords['x_pixels'] = self._data_array.coords['x_pixels']
        temp_array.coords['y_pixels'] = self._data_array.coords['y_pixels']
        temp_array.coords['Time of flight'] = self._data_array.coords['tof']
        temp_array = temp_array.rename_dims({'t': 'Time of flight'})

        if _is_notebook():
            return pp.inspector(temp_array, **inspector_kwargs_defaults)
        else:
            raise RuntimeError('Interactive spectrum inspector is only supported in Jupyter notebooks.')

    def roi_creator(self, **kwargs) -> None:
        """Launch an interactive ROI creator for defining regions of interest on the measurement data.

        This method uses the [plopp](https://scipp.github.io/plopp/plotting/roi-selector.html) library for interactive ROI
         creation.

        To customize the plot appearance, additional keyword arguments can be passed to the underlying plopp plotting function.
        See [plopp.inspector][] for which keyword arguments are available and how to use them.


        Interactive controls
        --------
        Click the ![](../../assets/images/vector-square.png){width=15px} button in the toolbar on the left to
         activate/deactivate the ROI creator tool. When active, the following controls are available:

        - Left-click to start drawing a new rectangular ROI, and left-click again to finish drawing the rectangle and create
         the ROI
        - Left-click and hold on ROI corners to resize the ROI
        - Right-click and hold to drag/move the entire ROI
        - Middle-click in the ROI to delete it

        Example
        ---------
        An example on how to use this method is shown in the tutorial notebook 

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the ROI creator function.
            See https://scipp.github.io/plopp/generated/plopp.inspector.html for options.

        Returns
        -------
        list
            A list of plopp figure objects comprising the ROI creator widget.
            Changes made interactively are reflected in :attr:`regions_of_interest`.

        Raises
        ------
        RuntimeError
            If called outside a Jupyter notebook environment.
        """

        if not _is_notebook():
            raise RuntimeError('Interactive ROI creator is only supported in Jupyter notebooks.')

        roi_selector_kwargs_defaults = self._plot_defaults()
        roi_selector_kwargs_defaults['ymax'] = roi_selector_kwargs_defaults['cmax']
        roi_selector_kwargs_defaults.update({
            'title': self.display_name + ' - ROI Creator',
            'ymin': 0.0,
            'dim': 'Time of flight',  # Due to https://github.com/scipp/plopp/issues/566
            'orientation': 'vertical',
            'operation': 'mean',
            'mode': 'rectangle',
            # 'autoscale' : False,
        })
        # Overwrite defaults with any user-provided kwargs
        roi_selector_kwargs_defaults.update(kwargs)

        # Hack to cirmunvent the bug in https://github.com/scipp/plopp/issues/566
        temp_array = self._data_array.drop_coords(self._data_array.coords)
        if self._has_physical_coords:
            temp_array.coords['x'] = self._data_array.coords['x']
            temp_array.coords['y'] = self._data_array.coords['y']
        temp_array.coords['x_pixels'] = self._data_array.coords['x_pixels']
        temp_array.coords['y_pixels'] = self._data_array.coords['y_pixels']
        temp_array.coords['Time of flight'] = self._data_array.coords['tof']
        temp_array = temp_array.rename_dims({'t': 'Time of flight'})

        plots = pp.inspector(temp_array, **roi_selector_kwargs_defaults)
        # plots = pp.inspector(self._data_array, **roi_selector_kwargs_defaults)

        # -------------------------------------------------------------------------------------------------
        # -------------------------- Plot the existing ROIs on the plot -----------------------------------
        # -------------------------------------------------------------------------------------------------

        for roi in self.regions_of_interest:
            if self._has_physical_coords and roi._has_physical_coords:
                x_start = roi.x_start.to(unit=self._data_array.coords['x'].unit).value
                y_start = roi.y_start.to(unit=self._data_array.coords['y'].unit).value
                x_end = roi.x_end.to(unit=self._data_array.coords['x'].unit).value
                y_end = roi.y_end.to(unit=self._data_array.coords['y'].unit).value
            elif self._has_physical_coords and not roi._has_physical_coords:
                x_slice, y_slice = roi.pixel_slice()
                sliced_array = self._data_array['x_pixels', x_slice]['y_pixels', y_slice]
                x_start = sliced_array.coords['x'].min().value
                y_start = sliced_array.coords['y'].min().value
                x_end = sliced_array.coords['x'].max().value
                y_end = sliced_array.coords['y'].max().value
            else:
                x_start = roi.x_pixel_start
                y_start = roi.y_pixel_start
                x_end = roi.x_pixel_end
                y_end = roi.y_pixel_end
            plots[0].toolbar['inspect']._tool.start()
            plots[0].toolbar['inspect']._tool.click(x=x_start, y=y_start, button=1)  # button 1 is left-click
            plots[0].toolbar['inspect']._tool.click(x=x_end, y=y_end, button=1)
            plots[0].toolbar['inspect']._tool.stop()
            if hasattr(roi, '_rect_ids'):
                roi._rect_ids.append(
                    plots[0].toolbar['inspect']._tool.children[-1].id
                )  # Store the rectangle ID for reference when dragging corners  # noqa: E501
            else:
                roi._rect_ids = [plots[0].toolbar['inspect']._tool.children[-1].id]

        # -------------------------------------------------------------------------------------------------
        # ------------- Define callbacks for creating, editing, and deleting ROIs -------------------------
        # -------------------------------------------------------------------------------------------------

        # The callback to be used by the Scipp RectangleTool when drawing a new rectangle.
        def create_rectangle_roi(rect, roi_list, data_array):
            """Create a new :class:`RectangleROI` from a drawn rectangle and append it to ``roi_list``.

            Parameters
            ----------
            rect :
                The rectangle object provided by the plopp RectangleTool.
            roi_list : EasyList[RectangleROI]
                The list of ROIs to append the new ROI to.
            data_array : sc.DataArray
                The data array used to derive pixel and physical coordinate ranges.
            """
            # Get the pixel and physical coordinate ranges from the rectangle vertices using the helper method.
            x_pixel_range, y_pixel_range, x_range, y_range = Measurement._ranges_from_rectangle(rect, data_array)
            new_roi = RectangleROI(
                x_pixel_range=x_pixel_range,
                y_pixel_range=y_pixel_range,
                x_range=x_range,
                y_range=y_range,
            )
            new_roi._rect_ids = [rect.id]  # Store the rectangle ID for reference when dragging corners
            roi_list.append(new_roi)

        # The callback to be used by the Scipp RectangleTool when dragging the corners of an existing rectangle.
        def edit_rectangle_roi(rect, roi_list, data_array):
            """Update the matching :class:`RectangleROI` in ``roi_list`` when a rectangle is resized.

            Parameters
            ----------
            rect :
                The rectangle object provided by the plopp RectangleTool.
            roi_list : EasyList[RectangleROI]
                The list of ROIs containing the ROI to update.
            data_array : sc.DataArray
                The data array used to derive updated coordinate ranges.
            """
            x_pixel_range, y_pixel_range, x_range, y_range = Measurement._ranges_from_rectangle(rect, data_array)
            # Find the ROI corresponding to the rectangle being edited based on the stored rectangle ID.
            for roi in roi_list:
                if hasattr(roi, '_rect_ids') and rect.id in roi._rect_ids:
                    matching_roi = roi
                    break

            matching_roi.set_pixel_coord_range(x_pixel_range, y_pixel_range)
            if x_range is not None:
                matching_roi.set_physical_coord_range(x_range, y_range)

        def delete_rectangle_roi(rect, roi_list):
            """Remove the :class:`RectangleROI` corresponding to ``rect`` from ``roi_list``.

            Parameters
            ----------
            rect :
                The rectangle object provided by the plopp RectangleTool.
            roi_list : EasyList[RectangleROI]
                The list of ROIs from which the matching ROI will be removed.
            """
            for roi in roi_list:
                if hasattr(roi, '_rect_ids') and rect.id in roi._rect_ids:
                    roi_list.remove(roi)
                    break

        # -------------------------------------------------------------------------------------------------
        # -------------------------- Connect the callbacks to the RectangleTool ---------------------------
        # -------------------------------------------------------------------------------------------------

        plots[0].toolbar['inspect']._tool.on_create(
            partial(
                create_rectangle_roi,
                roi_list=self.regions_of_interest,
                data_array=self._data_array,
            )
        )

        plots[0].toolbar['inspect']._tool.on_change(
            partial(edit_rectangle_roi, roi_list=self.regions_of_interest, data_array=self._data_array)
        )

        plots[0].toolbar['inspect']._tool.on_remove(partial(delete_rectangle_roi, roi_list=self.regions_of_interest))

        plots[0].toolbar['inspect'].tooltip = 'Activate ROI creator tool'

        return plots

    def spectrum(self, roi: RectangleROI | str | None = None) -> sc.DataArray:
        """Extract the spectrum (intensity vs. time-of-flight) for a specified region of interest (ROI).

        If no ROI is provided, the spectrum is calculated over the entire image.

        Parameters
        ----------
        roi : RectangleROI | str | None, optional
            The region of interest for which to extract the spectrum.
            If a string is provided, it should be the unique name of a predefined ROI in the measurement's list of ROIs.
            By default, None.

        Returns
        -------
        sc.DataArray
            A 1-D DataArray containing the spatially averaged spectrum along the
            time-of-flight axis.

        Raises
        ------
        TypeError
            If ``roi`` is not a :class:`RectangleROI`, string, or ``None``.
        KeyError
            If a string ``roi`` does not match any ROI in :attr:`regions_of_interest`.
        """
        if roi is not None and not isinstance(roi, (RectangleROI, str)):
            raise TypeError('roi must be a string, None, or an instance of RectangleROI.')

        if isinstance(roi, str):
            if roi in self.regions_of_interest:
                roi = self.regions_of_interest[roi]
            else:
                raise KeyError(
                    f"ROI with unique name '{roi}' not found in the measurement's list of ROIs: "
                    f'[{", ".join(item.unique_name for item in self.regions_of_interest)}].'
                )
        if roi is None:
            spectrum_data = self._data_array.mean(dim=['x', 'y'])
        elif self._has_physical_coords and roi._has_physical_coords:
            x_slice, y_slice = roi.slice()
            spectrum_data = self._data_array['x', x_slice]['y', y_slice].mean(dim=['x', 'y'])
        else:
            x_slice, y_slice = roi.pixel_slice()
            spectrum_data = self._data_array['x_pixels', x_slice]['y_pixels', y_slice].mean(dim=['x', 'y'])
        return spectrum_data

    def spectrum_plot(self, roi: RectangleROI | str | None = None, **kwargs) -> None:
        """Plot the spectrum (intensity vs. time-of-flight) for a specified region of interest (ROI).

        If no ROI is provided, the spectrum is calculated over the entire image.

        This method uses the plopp library for plotting:
        https://scipp.github.io/plopp/plotting/line-plot.html

        Parameters
        ----------
        roi : RectangleROI | str | None, optional
            The region of interest for which to plot the spectrum.
            If a string is provided, it should be the unique name of a predefined ROI in the measurement's list of ROIs.
            By default, None.
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function.
            See https://scipp.github.io/plopp/generated/plopp.plot.html for options.

        Returns
        -------
        plopp.Figure or None
            The plot object when running inside a Jupyter notebook, otherwise ``None``
            (the plot is displayed directly via :meth:`show`).

        Raises
        ------
        TypeError
            If ``roi`` is not a :class:`RectangleROI`, string, or ``None``.
        KeyError
            If a string ``roi`` does not match any ROI in :attr:`regions_of_interest`.
        """
        spectrum_data = self.spectrum(roi=roi)

        if roi is None:
            roi_name = 'entire image'
        elif isinstance(roi, str):
            roi_name = f"ROI '{roi}'"
        else:
            roi_name = f"ROI '{roi.unique_name}'"

        plot_kwargs_defaults = {
            'title': self.display_name + ' - Spectrum of ' + roi_name,
            'xlabel': f'Time of flight [{spectrum_data.coords["tof"].unit}]',
            'ylabel': 'Transmission',
            'ymin': 0.0,
            'ymax': min(3.0, float(spectrum_data.max().value * 1.1)),
            'coords': ['tof'],
        }
        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        if _is_notebook():
            return spectrum_data.plot(**plot_kwargs_defaults)
        else:
            plot = spectrum_data.plot(**plot_kwargs_defaults)
            plot.show()

    def _validate_data_array_coordinate(
        self,
        data_array: sc.DataArray,
        coord_name: str,
        expected_dim: str,
        expected_dim_string: str,
        expected_unit: str,
    ) -> None:
        """Validate that a named coordinate of a data array has the expected dimension and unit.

        Parameters
        ----------
        data_array : sc.DataArray
            The data array whose coordinate is to be validated.
        coord_name : str
            Name of the coordinate to validate.
        expected_dim : str
            The single dimension name the coordinate must have (e.g. ``'t'``).
        expected_dim_string : str
            Human-readable name of the physical dimension (e.g. ``'time'``).
        expected_unit : str
            The SI unit string the coordinate must be convertible to (e.g. ``'s'``).

        Raises
        ------
        DimensionError
            If the coordinate does not have exactly the expected dimension.
        UnitError
            If the coordinate cannot be converted to ``expected_unit``.
        """
        if data_array.coords[coord_name].dims != (expected_dim,):
            raise DimensionError(f"'{coord_name}' coordinate must be of dimension '{expected_dim}'.")
        try:
            data_array.coords[coord_name].to(unit=expected_unit)
        except UnitError:
            raise UnitError(
                f"'{coord_name}' coordinate must have a unit of {expected_dim_string}, such as ('{expected_unit}')."
            ) from None  # noqa: E501

    @staticmethod
    def _validate_provided_coord(
        data_array: sc.DataArray,
        coord: sc.Variable | np.ndarray,
        coord_name: str,
        dim: str,
        length_context: str,
        expected_dim_string: str,
        expected_unit: str,
    ) -> sc.Variable:
        """Validate and normalise a coordinate array provided by the caller.

        Accepts either a :class:`scipp.Variable` or a :class:`numpy.ndarray` and
        returns a validated :class:`scipp.Variable`. A numpy array is wrapped with
        ``dim`` and ``expected_unit``.

        Parameters
        ----------
        data_array : sc.DataArray
            The data array the coordinate will be assigned to (used for length checks).
        coord : sc.Variable | np.ndarray
            The coordinate values to validate.
        coord_name : str
            Human-readable name of the coordinate (used in error messages).
        dim : str
            Dimension name along which the coordinate runs (e.g. ``'t'``).
        length_context : str
            Description of what the length should match (used in error messages).
        expected_dim_string : str
            Human-readable name of the physical dimension (e.g. ``'time'``).
        expected_unit : str
            The SI unit string the coordinate must be convertible to (e.g. ``'s'``).

        Returns
        -------
        sc.Variable
            The validated (and possibly converted) coordinate variable.

        Raises
        ------
        TypeError
            If ``coord`` is not a :class:`scipp.Variable` array or :class:`numpy.ndarray`.
        ValueError
            If the length of ``coord`` does not match the dimension size of ``data_array``.
        UnitError
            If ``coord`` cannot be converted to ``expected_unit``.
        """
        if not (isinstance(coord, sc.Variable) and coord.sizes) and not isinstance(coord, np.ndarray):
            raise TypeError(f'{coord_name} must be a scipp Variable or a numpy ndarray.')
        if len(coord) not in (data_array.sizes[dim], data_array.sizes[dim] + 1):
            raise ValueError(f'Length of {coord_name} array does not match the number of {length_context}.')
        if isinstance(coord, np.ndarray):
            coord = sc.array(dims=[dim], values=coord, unit=expected_unit)
        try:
            coord.to(unit=expected_unit)
        except UnitError:
            raise UnitError(f"{coord_name} must have a unit of {expected_dim_string}, such as '{expected_unit}'.") from None
        return coord

    @staticmethod
    def _ranges_from_rectangle(
        rect, data_array: sc.DataArray
    ) -> tuple[tuple[int, int], None] | tuple[tuple[int, int], tuple[sc.Variable, sc.Variable]]:  # noqa: E501
        """Convert rectangle vertices from the plopp RectangleTool to pixel and physical coordinate ranges.

        Used internally by :meth:`roi_creator` when the user draws or edits a rectangle.

        Parameters
        ----------
        rect :
            Rectangle object from the plopp RectangleTool exposing a ``vertices`` attribute.
        data_array : sc.DataArray
            The current data array, used to determine whether physical coordinates are
            available and to look up pixel indices.

        Returns
        -------
        x_pixel_range : tuple[int, int]
            ``(x_start, x_end)`` pixel indices.
        y_pixel_range : tuple[int, int]
            ``(y_start, y_end)`` pixel indices.
        x_range : tuple[sc.Variable, sc.Variable] or None
            ``(x_start, x_end)`` physical coordinates, or ``None`` if no physical
            coordinates are present.
        y_range : tuple[sc.Variable, sc.Variable] or None
            ``(y_start, y_end)`` physical coordinates, or ``None`` if no physical
            coordinates are present.
        """
        # To be used in the roi_selector method to convert the rectangle vertices to pixel and
        # physical coordinate ranges for the new ROI.
        x_vertex_list, y_vertex_list = rect.vertices
        if 'x' not in data_array.coords:  # If 'x' exists, so does 'y' due to our constructor.
            x_pixel_range = (int(min(x_vertex_list)), int(max(x_vertex_list)))
            y_pixel_range = (int(min(y_vertex_list)), int(max(y_vertex_list)))
            x_range = None
            y_range = None
        # Otherwise they're physical coordinates that we need to convert to pixel indices for the ROI.
        else:
            x_unit = data_array.coords['x'].unit
            y_unit = data_array.coords['y'].unit
            x_range = (
                sc.scalar(min(x_vertex_list), unit=x_unit),
                sc.scalar(max(x_vertex_list), unit=x_unit),
            )
            y_range = (
                sc.scalar(min(y_vertex_list), unit=y_unit),
                sc.scalar(max(y_vertex_list), unit=y_unit),
            )

            sliced_data_array = data_array['x', x_range[0] : x_range[1]]['y', y_range[0] : y_range[1]]

            sliced_x_pixels = sliced_data_array.coords['x_pixels'].values
            sliced_y_pixels = sliced_data_array.coords['y_pixels'].values
            x_pixel_range = (int(min(sliced_x_pixels)), int(max(sliced_x_pixels)))
            y_pixel_range = (int(min(sliced_y_pixels)), int(max(sliced_y_pixels)))

        return x_pixel_range, y_pixel_range, x_range, y_range

    def _plot_defaults(self):
        """Return the default keyword arguments shared across all plot methods.

        Returns
        -------
        dict
            A dictionary of default plopp keyword arguments including title, colour-bar
            label and limits, mask colour, and the coordinate names to use for the axes.
        """
        return {
            'title': self.display_name,
            'clabel': 'Transmission',
            'cmin': 0.0,
            'cmax': min(3.0, float(self._data_array.max().value * 1.1)),
            'mask_color': 'red',
            'coords': ['x_pixels', 'y_pixels'] if not self._has_physical_coords else ['x', 'y'],
        }

    def __repr__(self):
        """Return a string representation of the Measurement.

        Returns
        -------
        str
            A human-readable summary including the display name, data shape, and the
            display names of all associated regions of interest.
        """
        return f'{self.display_name} with shape {self._data_array.shape} and regions of interest: {[roi.display_name for roi in self.regions_of_interest]}'  # noqa: E501
