#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from typing import Sequence

import scipp as sc
from easyscience.base_classes import NewBase
from scipp import UnitError


class RectROI(NewBase):
    """Class representing a rectangular region of interest (ROI) in an image.

    Physical coordinate ranges are used by default if they are provided and are usable in the Measurement.
    Otherwise, the obligatory pixel coordinate ranges are used as fallback.
    """

    def __init__(
        self,
        x_pixel_range: tuple[int, int],
        y_pixel_range: tuple[int, int],
        x_range: tuple[sc.Variable, sc.Variable] | None = None,
        y_range: tuple[sc.Variable, sc.Variable] | None = None,
        unique_name: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """
        Initialize a RectROI instance.

        Parameters:
        x_pixel_range (tuple[int, int]): The pixel range for the x-coordinate.
        y_pixel_range (tuple[int, int]): The pixel range for the y-coordinate.
        x_range (tuple[sc.Variable, sc.Variable], optional): The physical coordinate range for the x-axis.
        y_range (tuple[sc.Variable, sc.Variable], optional): The physical coordinate range for the y-axis.
        unique_name (str | None, optional): A unique identifier for the ROI. Defaults to RectROI appended by a unique integer.
        display_name (str | None, optional): A pretty name for the ROI. Defaults to the unique_name if not provided.
        """
        super().__init__(unique_name=unique_name, display_name=display_name)
        self.set_pixel_coord_range(x_pixel_range, y_pixel_range)

        if x_range is not None and y_range is not None:
            self.set_physical_coord_range(x_range, y_range)
        elif x_range is None and y_range is None:
            self._has_physical_coords = False
        else:
            raise ValueError('Both x_range and y_range must be provided together or not at all.')

    def set_pixel_coord_range(self, x_pixel_range: Sequence[int, int], y_pixel_range: Sequence[int, int]) -> None:
        """Set the pixel coordinate ranges for the ROI.
        Parameters:
            x_pixel_range (tuple[int, int]): The start and end pixel coordinates in the x direction.
            y_pixel_range (tuple[int, int]): The start and end pixel coordinates in the y direction.
        """
        self._check_input_sequence(x_pixel_range, 'x_pixel_range', 'integers', int)
        self._check_input_sequence(y_pixel_range, 'y_pixel_range', 'integers', int)
        for index in tuple(x_pixel_range) + tuple(y_pixel_range):
            if index < 0:
                raise ValueError('Pixel indices must be non-negative integers.')  # Do I need this check?
        self._x_pixel_start, self._x_pixel_end = x_pixel_range
        self._y_pixel_start, self._y_pixel_end = y_pixel_range

    def set_physical_coord_range(
        self, x_range: Sequence[sc.Variable, sc.Variable], y_range: Sequence[sc.Variable, sc.Variable]
    ) -> None:
        """Set the physical coordinate ranges for the ROI.
        Parameters:
            x_range (tuple[sc.Variable, sc.Variable]): The start and end physical coordinates in the x direction.
            y_range (tuple[sc.Variable, sc.Variable]): The start and end physical coordinates in the y direction.
        """

        self._check_input_sequence(x_range, 'x_range', 'scipp scalars', sc.Variable)
        self._check_input_sequence(y_range, 'y_range', 'scipp scalars', sc.Variable)
        for scalar in tuple(x_range) + tuple(y_range):
            self._check_scalar(scalar, 'Physical coordinates')
        self._x_start, self._x_end = x_range
        self._y_start, self._y_end = y_range
        self._has_physical_coords = True

    def delete_physical_coord_range(self) -> None:
        """Delete the physical coordinate ranges for the ROI."""
        if self._has_physical_coords:
            del self._x_start
            del self._x_end
            del self._y_start
            del self._y_end
            self._has_physical_coords = False
        else:
            raise ValueError('Cannot delete physical coordinate ranges because they are not set.')

    @property
    def x_pixel_start(self) -> int:
        return self._x_pixel_start

    @x_pixel_start.setter
    def x_pixel_start(self, value: int):
        self._check_index(value, 'x_pixel_start')
        self._x_pixel_start = value

    @property
    def x_pixel_end(self) -> int:
        return self._x_pixel_end

    @x_pixel_end.setter
    def x_pixel_end(self, value: int):
        self._check_index(value, 'x_pixel_end')
        self._x_pixel_end = value

    @property
    def y_pixel_start(self) -> int:
        return self._y_pixel_start

    @y_pixel_start.setter
    def y_pixel_start(self, value: int):
        self._check_index(value, 'y_pixel_start')
        self._y_pixel_start = value

    @property
    def y_pixel_end(self) -> int:
        return self._y_pixel_end

    @y_pixel_end.setter
    def y_pixel_end(self, value: int):
        self._check_index(value, 'y_pixel_end')
        self._y_pixel_end = value

    @property
    def x_start(self) -> sc.Variable:
        return self._x_start.copy()

    @x_start.setter
    def x_start(self, value: sc.Variable):
        self._single_coord_setter_check(value, 'x_start')
        self._check_scalar(value, 'x_start')
        self._x_start = value

    @property
    def x_end(self) -> sc.Variable:
        return self._x_end.copy()

    @x_end.setter
    def x_end(self, value: sc.Variable):
        self._single_coord_setter_check(value, 'x_end')
        self._check_scalar(value, 'x_end')
        self._x_end = value

    @property
    def y_start(self) -> sc.Variable:
        return self._y_start.copy()

    @y_start.setter
    def y_start(self, value: sc.Variable):
        self._single_coord_setter_check(value, 'y_start')
        self._check_scalar(value, 'y_start')
        self._y_start = value

    @property
    def y_end(self) -> sc.Variable:
        return self._y_end.copy()

    @y_end.setter
    def y_end(self, value: sc.Variable):
        self._single_coord_setter_check(value, 'y_end')
        self._check_scalar(value, 'y_end')
        self._y_end = value

    def _check_input_sequence(self, value: Sequence, name: str, typename: str, expected_type: any) -> None:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise TypeError(f'{name} must be a tuple or a list of two {typename}, got a {type(value).__name__}.')
        if not (isinstance(value[0], expected_type) and isinstance(value[1], expected_type)):
            raise TypeError(
                f'{name} must be a tuple or a list of two {typename}, '
                f'got {type(value[0]).__name__} and {type(value[1]).__name__}.'
            )

    def _check_index(self, value: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f'{name} indice must be an integer.')
        if value < 0:
            raise ValueError(f'{name} indice must be non-negative.')

    def _check_scalar(self, value: sc.Variable, name: str) -> None:
        if value.dims:
            raise ValueError(f'{name} must be a scipp scalar (0-dimensional Variable).')
        try:
            test_value = value.copy()
            test_value.to(unit='m')
        except Exception as e:
            raise UnitError(f"{name} must be a scipp scalar with a unit of length (e.g., 'm').") from e

    def _single_coord_setter_check(self, value: sc.Variable, name: str) -> None:
        if not isinstance(value, sc.Variable):
            raise TypeError(f'{name} must be a scipp scalar.')
        if not self._has_physical_coords:
            raise ValueError(
                f'Cannot set {name} before setting all physical coordinate ranges. '
                'Please use the set_physical_coord_range method.'
            )

    def __repr__(self) -> str:
        repr_str = (
            f'RectROI(x_pixel_range=({self._x_pixel_start}, {self._x_pixel_end}), '
            f'y_pixel_range=({self._y_pixel_start}, {self._y_pixel_end})'
        )
        if self._has_physical_coords:
            repr_str += f', x_range=({self._x_start}, {self._x_end}), y_range=({self._y_start}, {self._y_end})'
        repr_str += ')'
        return repr_str
