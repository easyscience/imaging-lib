#  SPDX-FileCopyrightText: 2026 EasyImaging contributors  <imaging@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2026 Contributors to the EasyImaging project <https://github.com/easyScience/EasyImaging>
from __future__ import annotations

from easyscience.base_classes import NewBase
import scipp as sc

class RectROI(NewBase):
    """Class representing a rectangular region of interest (ROI) in an image."""

    def __init__(self, 
                 x_pixels: tuple[int, int], 
                 y_pixels: tuple[int, int], 
                 x: tuple[sc.Variable, sc.Variable] = None, 
                 y: tuple[sc.Variable, sc.Variable] = None,
                 ) -> None:
        """
        Initialize a RectROI instance.

        Parameters:
        x_pixels (tuple[int, int]): The pixel range for the x-coordinate.
        y_pixels (tuple[int, int]): The pixel range for the y-coordinate.
        x (tuple[sc.Variable, sc.Variable], optional): The physical coordinate range for the x-axis.
        y (tuple[sc.Variable, sc.Variable], optional): The physical coordinate range for the y-axis.
        """
        self._check_input_tuple(x_pixels, "x_pixels", "integers", int)
        self._check_input_tuple(y_pixels, "y_pixels", "integers", int)
        for index in x_pixels + y_pixels:
            if index < 0:
                raise ValueError("Pixel indices must be non-negative integers.")
        self._x_pixel_start, self._x_pixel_end = x_pixels
        self._y_pixel_start, self._y_pixel_end = y_pixels

        if x is not None & y is not None:
            self._check_input_tuple(x, "x physical coordinates", "scipp scalars", sc.Variable)
            self._check_input_tuple(y, "y physical coordinates", "scipp scalars", sc.Variable)
            for scalar in x + y:
                if scalar.dims:
                    raise ValueError("Physical coordinates must be a scipp scalar (0-dimensional Variable).")
                try:
                    test_value = scalar.copy()
                    test_value.to('m')
                except Exception:
                    raise TypeError("Physical coordinates must be a scipp scalar with a unit of length (e.g., 'm').")
            self._x_start, self._x_end = x
            self._y_start, self._y_end = y
        elif x is None and y is None:
            pass
        else:    
            raise ValueError("Both x and y physical coordinates must be provided together.")

    @property
    def x_pixel_start(self) -> int:
        return(self._x_pixel_start)
    
    @x_pixel_start.setter
    def x_pixel_start(self, value: int):
        self._check_index(value, "x_pixel_start")
        self._x_pixel_start = value

    @property
    def x_pixel_end(self) -> int:
        return(self._x_pixel_end)
    
    @x_pixel_end.setter
    def x_pixel_end(self, value: int):
        self._check_index(value, "x_pixel_end")
        self._x_pixel_end = value

    @property
    def y_pixel_start(self) -> int:
        return(self._y_pixel_start)
    
    @y_pixel_start.setter
    def y_pixel_start(self, value: int):
        self._check_index(value, "y_pixel_start")
        self._y_pixel_start = value

    @property
    def y_pixel_end(self) -> int:
        return(self._y_pixel_end)
    
    @y_pixel_end.setter
    def y_pixel_end(self, value: int):
        self._check_index(value, "y_pixel_end")
        self._y_pixel_end = value

    @property
    def x_start(self) -> sc.Variable:
        return(self._x_start.copy())
    
    @x_start.setter
    def x_start(self, value: sc.Variable):
        self._check_scalar(value, "x_start")
        self._x_start = value

    @property
    def x_end(self) -> sc.Variable:
        return(self._x_end.copy())
    
    @x_end.setter
    def x_end(self, value: sc.Variable):
        self._check_scalar(value, "x_end")
        self._x_end = value

    @property
    def y_start(self) -> sc.Variable:
        return(self._y_start.copy())
    
    @y_start.setter
    def y_start(self, value: sc.Variable):
        self._check_scalar(value, "y_start")
        self._y_start = value

    @property
    def y_end(self) -> sc.Variable:
        return(self._y_end.copy())
    
    @y_end.setter
    def y_end(self, value: sc.Variable):
        self._check_scalar(value, "y_end")
        self._y_end = value

    def _check_input_tuple(self, value: tuple[any, any], name: str, typename: str, type: any) -> None:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} must be a tuple of two {typename}.")
        if not (isinstance(value[0], type) and isinstance(value[1], type)):
            raise TypeError(f"{name} must be a tuple of two {typename}.")
        
    def _check_index(self, value: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{name} indice must be an integer.")
        if value < 0:
            raise ValueError(f"{name} indice must be non-negative.")
    
    def _check_scalar(self, value: sc.Variable, name: str) -> None:
        if not isinstance(value, sc.Variable):
            raise TypeError(f"{name} must be a scipp scalar.")
        if value.dims:
            raise ValueError(f"{name} must be a scipp scalar (0-dimensional Variable).")
        try:
            test_value = value.copy()
            test_value.to('m')
        except Exception:
            raise ValueError(f"{name} must be a scipp scalar with a unit of length (e.g., 'm').")