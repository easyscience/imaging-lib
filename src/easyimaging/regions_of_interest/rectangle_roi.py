# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import scipp as sc
from easyscience.base_classes import NewBase
from scipp import UnitError


class RectangleROI(NewBase):
    """Object representing a rectangular region of interest (ROI) in an image.

    This object is most simply created by the [`roi_creator`][...Measurement.roi_creator] method in the
      [Measurement][...Measurement] class. Alternatively it can be created manually as shown in the example below.

    Physical coordinate ranges are used by default, if they are provided **and** can be used in the
      associated [Measurement][...Measurement] class.
    Otherwise, the obligatory pixel coordinate ranges are used as a fallback.

    Example
    -------
    Creating a [RectangleROI][.] instance manually:
    ```python
    import scipp as sc
    from easyimaging.regions_of_interest import RectangleROI

    roi = RectangleROI(
        x_pixel_range=(10, 50),
        y_pixel_range=(20, 80),
        x_range=(sc.scalar(0.0, unit='m'), sc.scalar(10.0, unit='m')),
        y_range=(sc.scalar(0.0, unit='m'), sc.scalar(5.0, unit='m')),
        unique_name='my_rectangle_roi',
    )
    ```
    """

    def __init__(
        self,
        x_pixel_range: Sequence[int],
        y_pixel_range: Sequence[int],
        x_range: Sequence[sc.Variable] | None = None,
        y_range: Sequence[sc.Variable] | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Initialize a RectangleROI instance.

        Parameters
        ----------
        x_pixel_range : Sequence[int]
            A two-element integer sequence ``[x_start, x_end]`` indicating the pixel range in the
            x-direction.
        y_pixel_range : Sequence[int]
            A two-element integer sequence ``[y_start, y_end]`` indicating the pixel range in the
            y-direction.
        x_range : Sequence[sc.Variable] | None
            A two-element sequence ``[x_start, x_end]`` of [`sc.Variable`][scipp.Variable] scalars with units of length,
             indicating the spatial coordinate range in the x-direction.<br>
            Must be provided together with ``y_range``.
        y_range : Sequence[sc.Variable] | None
            A two-element sequence ``[y_start, y_end]`` of [`sc.Variable`][scipp.Variable] scalars with units of length,
             indicating the spatial coordinate range in the y-direction.<br>
            Must be provided together with ``x_range``.
        unique_name : str | None
            A unique identifier for the ROI. Defaults to ``'RectangleROI'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the ROI. Defaults to [`unique_name`][..unique_name] if not provided.

        Raises
        ------
        TypeError
            If ``x_pixel_range`` or ``y_pixel_range`` is not a two-element sequence of integers.<br>
            If physical ranges are not two-element sequences of [`sc.Variable`][scipp.Variable].
        ValueError
            If any pixel index in ``x_pixel_range`` or ``y_pixel_range`` is negative.<br>
            If only one of ``x_range`` / ``y_range`` is provided.<br>
            If any [`sc.Variable`][scipp.Variable] in ``x_range`` or ``y_range`` is not 0-dimensional (i.e., not a scalar).<br>
        UnitError
            If any [`sc.Variable`][scipp.Variable] scalar in ``x_range`` or ``y_range`` does not have a unit of length.
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        self.set_pixel_coord_range(x_pixel_range, y_pixel_range)

        if x_range is not None and y_range is not None:
            self.set_physical_coord_range(x_range, y_range)
        elif x_range is None and y_range is None:
            self._has_physical_coords = False
        else:
            raise ValueError('Both x_range and y_range must be provided together or not at all.')

    def set_pixel_coord_range(self, x_pixel_range: Sequence[int], y_pixel_range: Sequence[int]) -> None:
        """Set the pixel coordinate ranges defining the ROI.

        Parameters
        ----------
        x_pixel_range : Sequence[int]
            The pixel coordinate range in the x-direction to set.
            Must be a two-element sequence ``[x_start, x_end]`` of pixel indices defining an x-axis range.
        y_pixel_range : Sequence[int]
            The pixel coordinate range in the y-direction to set.
            Must be a two-element sequence ``[y_start, y_end]`` of pixel indices defining a y-axis range.

        Raises
        ------
        TypeError
            If either argument is not a two-element sequence of integers.
        ValueError
            If any pixel index is negative.
        """
        self._check_input_sequence(x_pixel_range, 'x_pixel_range', 'integers', int)
        self._check_input_sequence(y_pixel_range, 'y_pixel_range', 'integers', int)
        for index in tuple(x_pixel_range) + tuple(y_pixel_range):
            if index < 0:
                raise ValueError('Pixel indices must be non-negative integers.')
        self._x_pixel_start, self._x_pixel_end = sc.array(values=x_pixel_range, dims='x')
        self._y_pixel_start, self._y_pixel_end = sc.array(values=y_pixel_range, dims='y')

    def set_physical_coord_range(self, x_range: Sequence[sc.Variable], y_range: Sequence[sc.Variable]) -> None:
        """Set the physical coordinate ranges defining the ROI.

        Parameters
        ----------
        x_range : Sequence[sc.Variable]
            The physical coordinate range in the x-direction to set.
            Must be a two-element sequence ``[x_start, x_end]`` of [`sc.Variable`][scipp.Variable] scalars with units of length
        y_range : Sequence[sc.Variable]
            The physical coordinate range in the y-direction to set.
            Must be a two-element sequence ``[y_start, y_end]`` of [`sc.Variable`][scipp.Variable] scalars with units of length

        Raises
        ------
        TypeError
            If either argument is not a two-element sequence of [`sc.Variable`][scipp.Variable].
        ValueError
            If any [`sc.Variable`][scipp.Variable] in ``x_range`` or ``y_range`` is not 0-dimensional (i.e., not a scalar).
        UnitError
            If any [`sc.Variable`][scipp.Variable] scalar in ``x_range`` or ``y_range`` does not have a unit of length.
        """

        self._check_input_sequence(x_range, 'x_range', 'scipp scalars', sc.Variable)
        self._check_input_sequence(y_range, 'y_range', 'scipp scalars', sc.Variable)
        for scalar in tuple(x_range) + tuple(y_range):
            self._check_scalar(scalar, 'Physical coordinates')
        self._x_start, self._x_end = x_range
        self._y_start, self._y_end = y_range
        self._has_physical_coords = True

    def delete_physical_coord_range(self) -> None:
        """Delete the physical coordinate ranges for the ROI.

        This enforces the ROI to use pixel coordinates only.

        Raises
        ------
        ValueError
            If physical coordinate ranges are not currently set.
        """
        if self._has_physical_coords:
            del self._x_start
            del self._x_end
            del self._y_start
            del self._y_end
            self._has_physical_coords = False
        else:
            raise ValueError('Cannot delete physical coordinate ranges because they are not set.')

    def pixel_slice(self) -> Tuple[slice, slice]:
        """Get a pixel index slice object corresponding to the ROI.

        Returns
        -------
        tuple
            Two (x,y) slice objects representing the pixel ranges of the ROI.
        """
        return slice(self._x_pixel_start, self._x_pixel_end), slice(self._y_pixel_start, self._y_pixel_end)

    def slice(self) -> Tuple[slice, slice]:
        """Get a physical coordinate slice object corresponding to the ROI.

        Returns
        -------
        tuple
            Two (x,y) slice objects representing the physical coordinate ranges of the ROI.

        Raises
        ------
        ValueError
            If physical coordinate ranges are not set for this ROI.
        """
        if self._has_physical_coords:
            return slice(self._x_start, self._x_end), slice(self._y_start, self._y_end)
        else:
            raise ValueError('Physical coordinate ranges are not set for this ROI.')

    @property
    def x_pixel_start(self) -> int:
        """Pixel index defining the start of the ROI in the x-direction.

        Parameters
        ----------
        value : int
            A non-negative integer pixel index.

        Returns
        -------
        int
            The x-axis pixel index defining the start of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not an integer.
        ValueError
            If ``value`` is negative.
        """
        return self._x_pixel_start.value

    @x_pixel_start.setter
    def x_pixel_start(self, value: int):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._check_index(value, 'x_pixel_start')
        self._x_pixel_start = sc.scalar(value)

    @property
    def x_pixel_end(self) -> int:
        """Pixel index defining the end of the ROI in the x-direction.

        Parameters
        ----------
        value : int
            A non-negative integer pixel index.

        Returns
        -------
        int
            The x-axis pixel index defining the end of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not an integer.
        ValueError
            If ``value`` is negative.
        """
        return self._x_pixel_end.value

    @x_pixel_end.setter
    def x_pixel_end(self, value: int):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._check_index(value, 'x_pixel_end')
        self._x_pixel_end = sc.scalar(value)

    @property
    def y_pixel_start(self) -> int:
        """Pixel index defining the start of the ROI in the y-direction.

        Parameters
        ----------
        value : int
            A non-negative integer pixel index.

        Returns
        -------
        int
            The y-axis pixel index defining the start of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not an integer.
        ValueError
            If ``value`` is negative.
        """
        return self._y_pixel_start.value

    @y_pixel_start.setter
    def y_pixel_start(self, value: int):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._check_index(value, 'y_pixel_start')
        self._y_pixel_start = sc.scalar(value)

    @property
    def y_pixel_end(self) -> int:
        """Pixel index defining the end of the ROI in the y-direction.

        Parameters
        ----------
        value : int
            A non-negative integer pixel index.

        Returns
        -------
        int
            The y-axis pixel index defining the end of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not an integer.
        ValueError
            If ``value`` is negative.
        """
        return self._y_pixel_end.value

    @y_pixel_end.setter
    def y_pixel_end(self, value: int):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._check_index(value, 'y_pixel_end')
        self._y_pixel_end = sc.scalar(value)

    @property
    def x_start(self) -> sc.Variable:
        """Physical coordinate defining the start of the ROI in the x-direction.

        Parameters
        ----------
        value : sc.Variable
            A 0-dimensional [`sc.Variable`][scipp.Variable] scalar with a unit
             of length.

        Returns
        -------
        sc.Variable
            A copy of the x-axis physical coordinate defining the start of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable].
        ValueError
            If physical coordinate ranges are not set (use
            [`set_physical_coord_range`][..set_physical_coord_range] to set).<br>
            If ``value`` is not a 0-dimensional [`sc.Variable`][scipp.Variable] (i.e., a scalar).
        UnitError
            If ``value`` does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._x_start.copy()
        else:
            raise ValueError('Physical coordinate ranges are not set for this ROI.')

    @x_start.setter
    def x_start(self, value: sc.Variable):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._single_coord_setter_check(value, 'x_start')
        self._check_scalar(value, 'x_start')
        self._x_start = value

    @property
    def x_end(self) -> sc.Variable:
        """Physical coordinate defining the end of the ROI in the x-direction.

        Parameters
        ----------
        value : sc.Variable
            A 0-dimensional [`sc.Variable`][scipp.Variable] scalar with a unit of length.

        Returns
        -------
        sc.Variable
            A copy of the x-axis physical coordinate defining the end of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable].
        ValueError
            If physical coordinate ranges are not set (use
            [`set_physical_coord_range`][..set_physical_coord_range] to set).<br>
            If ``value`` is not a 0-dimensional [`sc.Variable`][scipp.Variable] (i.e., a scalar).
        UnitError
            If ``value`` does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._x_end.copy()
        else:
            raise ValueError('Physical coordinate ranges are not set for this ROI.')

    @x_end.setter
    def x_end(self, value: sc.Variable):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._single_coord_setter_check(value, 'x_end')
        self._check_scalar(value, 'x_end')
        self._x_end = value

    @property
    def y_start(self) -> sc.Variable:
        """Physical coordinate defining the start of the ROI in the y-direction.

        Parameters
        ----------
        value : sc.Variable
            A 0-dimensional [`sc.Variable`][scipp.Variable] scalar with a unit
             of length.

        Returns
        -------
        sc.Variable
            A copy of the y-axis physical coordinate defining the start of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable].
        ValueError
            If physical coordinate ranges are not set (use
            [`set_physical_coord_range`][..set_physical_coord_range] to set).<br>
            If ``value`` is not a 0-dimensional [`sc.Variable`][scipp.Variable] (i.e., a scalar).
        UnitError
            If ``value`` does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._y_start.copy()
        else:
            raise ValueError('Physical coordinate ranges are not set for this ROI.')

    @y_start.setter
    def y_start(self, value: sc.Variable):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._single_coord_setter_check(value, 'y_start')
        self._check_scalar(value, 'y_start')
        self._y_start = value

    @property
    def y_end(self) -> sc.Variable:
        """Physical coordinate defining the end of the ROI in the y-direction.

        Parameters
        ----------
        value : sc.Variable
            A 0-dimensional [`sc.Variable`][scipp.Variable] scalar with a unit
             of length.

        Returns
        -------
        sc.Variable
            A copy of the y-axis physical coordinate defining the end of the ROI.

        Raises
        ------
        TypeError
            If ``value`` is not a [`sc.Variable`][scipp.Variable].
        ValueError
            If physical coordinate ranges are not set (use
            [`set_physical_coord_range`][..set_physical_coord_range] to set).<br>
            If ``value`` is not a 0-dimensional [`sc.Variable`][scipp.Variable] (i.e., a scalar).
        UnitError
            If ``value`` does not have a unit of length.
        """
        if self._has_physical_coords:
            return self._y_end.copy()
        else:
            raise ValueError('Physical coordinate ranges are not set for this ROI.')

    @y_end.setter
    def y_end(self, value: sc.Variable):
        # Setters have no docstrings. They should be written in the getter docstring instead.
        self._single_coord_setter_check(value, 'y_end')
        self._check_scalar(value, 'y_end')
        self._y_end = value

    def _check_input_sequence(self, value: Sequence, name: str, typename: str, expected_type: any) -> None:
        """Validate that ``value`` is a two-element sequence of the expected type.

        Parameters
        ----------
        value : Sequence
            The value to validate.
        name : str
            Name of the parameter (used in error messages).
        typename : str
            Human-readable name of the expected element type (used in error messages).
        expected_type : type
            The type each element must be an instance of.

        Raises
        ------
        TypeError
            If ``value`` is not a two-element sequence or its elements are not of
            ``expected_type``.
        """
        if not isinstance(value, Sequence) or len(value) != 2:
            raise TypeError(f'{name} must be a tuple or a list of two {typename}, got a {type(value).__name__}.')
        if not (isinstance(value[0], expected_type) and isinstance(value[1], expected_type)):
            raise TypeError(
                f'{name} must be a tuple or a list of two {typename}, '
                f'got {type(value[0]).__name__} and {type(value[1]).__name__}.'
            )

    def _check_index(self, value: int, name: str) -> None:
        """Validate that ``value`` is a non-negative integer suitable as a pixel index.

        Parameters
        ----------
        value : int
            The value to validate.
        name : str
            Name of the parameter (used in error messages).

        Raises
        ------
        TypeError
            If ``value`` is not an integer.
        ValueError
            If ``value`` is negative.
        """
        if not isinstance(value, int):
            raise TypeError(f'{name} index must be an integer.')
        if value < 0:
            raise ValueError(f'{name} index must be non-negative.')

    def _check_scalar(self, value: sc.Variable, name: str) -> None:
        """Validate that ``value`` is a 0-dimensional :class:`scipp.Variable` with a unit of length.

        Parameters
        ----------
        value : sc.Variable
            The value to validate.
        name : str
            Name of the parameter (used in error messages).

        Raises
        ------
        ValueError
            If ``value`` has dimensions (i.e. is not a scalar).
        UnitError
            If ``value`` cannot be converted to metres.
        """
        if value.dims:
            raise ValueError(f'{name} must be a scipp scalar (0-dimensional Variable).')
        try:
            test_value = value.copy()
            test_value.to(unit='m')
        except Exception as e:
            raise UnitError(f"{name} must be a scipp scalar with a unit of length (e.g., 'm').") from e

    def _single_coord_setter_check(self, value: sc.Variable, name: str) -> None:
        """Validate preconditions for setting a single physical coordinate scalar.

        Checks that ``value`` is a :class:`scipp.Variable` and that physical coordinate
        ranges have already been initialised.

        Parameters
        ----------
        value : sc.Variable
            The candidate value to set.
        name : str
            Name of the coordinate being set (used in error messages).

        Raises
        ------
        TypeError
            If ``value`` is not a :class:`scipp.Variable`.
        ValueError
            If physical coordinate ranges have not been set yet.
        """
        if not isinstance(value, sc.Variable):
            raise TypeError(f'{name} must be a scipp scalar.')
        if not self._has_physical_coords:
            raise ValueError(
                f'Cannot set {name} before setting all physical coordinate ranges. '
                'Please use the set_physical_coord_range method.'
            )

    def to_dict(self, skip: List[str] | None = None) -> Dict[str, Any]:
        """Convert the ROI instance to a dictionary representation.

        Used for saving the ROI to file.

        Parameters
        ----------
        skip : list[str] | None
            A list of attribute names to exclude from the output dictionary.<br>
            The ``'x_pixel_range'``, ``'y_pixel_range'``, ``'x_range'``, and
            ``'y_range'`` keys are always handled explicitly and appended
            regardless of this parameter.

        Returns
        -------
        dict
            A dictionary containing the serialised ROI data, including pixel ranges
            and, if available, physical coordinate ranges encoded as scipp scalar dicts.
        """
        if skip is None:
            skip = []
        elif isinstance(skip, str):
            skip = [skip]
        skip = list(skip)  # Create a copy of the skip list to avoid modifying the original
        skip.extend(['x_pixel_range', 'y_pixel_range', 'x_range', 'y_range'])
        out_dict = super().to_dict(skip=skip)
        out_dict['x_pixel_range'] = [int(self.x_pixel_start), int(self.x_pixel_end)]
        out_dict['y_pixel_range'] = [int(self.y_pixel_start), int(self.y_pixel_end)]
        if self._has_physical_coords:
            out_dict['x_range'] = [
                {
                    '@module': 'scipp',
                    '@version': sc.__version__,
                    '@class': 'scalar',
                    'dict': sc.to_dict(self.x_start),
                },
                {
                    '@module': 'scipp',
                    '@version': sc.__version__,
                    '@class': 'scalar',
                    'dict': sc.to_dict(self.x_end),
                },
            ]
            out_dict['y_range'] = [
                {
                    '@module': 'scipp',
                    '@version': sc.__version__,
                    '@class': 'scalar',
                    'dict': sc.to_dict(self.y_start),
                },
                {
                    '@module': 'scipp',
                    '@version': sc.__version__,
                    '@class': 'scalar',
                    'dict': sc.to_dict(self.y_end),
                },
            ]
        return out_dict

    @classmethod
    def from_dict(cls, input_dict: Dict[str, Any]) -> RectangleROI:
        """Create a [`RectangleROI`][..] instance from a dictionary representation.

        Parameters
        ----------
        input_dict : dict
            A dictionary in a format as produced by [`to_dict`][..to_dict], containing at minimum
            ``'x_pixel_range'`` and ``'y_pixel_range'`` keys.

        Returns
        -------
        RectangleROI
            A new [`RectangleROI`][..] instance initialised with the data from the dictionary.
        """
        temp_dict = input_dict.copy()
        if 'x_range' in input_dict and 'y_range' in input_dict:
            temp_dict['x_range'] = [sc.from_dict(item['dict']) for item in input_dict['x_range']]
            temp_dict['y_range'] = [sc.from_dict(item['dict']) for item in input_dict['y_range']]
        return super().from_dict(temp_dict)

    def __repr__(self) -> str:
        """Return a string representation of the RectangleROI.

        Returns
        -------
        str
            A human-readable summary including pixel ranges and, when available,
            physical coordinate ranges.
        """
        repr_str = (
            f'RectangleROI(x_pixel_range=({self._x_pixel_start}, {self._x_pixel_end}), '
            f'y_pixel_range=({self._y_pixel_start}, {self._y_pixel_end})'
        )
        if self._has_physical_coords:
            repr_str += f', x_range=({self._x_start}, {self._x_end}), y_range=({self._y_start}, {self._y_end})'
        repr_str += ')'
        return repr_str
