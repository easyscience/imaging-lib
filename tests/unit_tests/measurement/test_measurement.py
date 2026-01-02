from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from easyimaging import Measurement


class TestMeasurement:
    @pytest.fixture
    def valid_data_array(self):
        tof = sc.arange('t', 0, 10, 1, unit='s')
        x = sc.arange('x', 0, 5, 1, unit='m')
        y = sc.arange('y', 0, 5, 1, unit='m')
        data = sc.zeros(dims=('x', 'y', 't'), shape=(5, 5, 10))
        return sc.DataArray(data=data, coords={'tof': tof, 'x': x, 'y': y})

    def test_init_valid_data_array(self, valid_data_array):
        # When Then
        measurement = Measurement(
            data_array=valid_data_array,
            unique_name="test_measurement",
            display_name="Test Measurement"
            )
        # Expect
        assert sc.identical(measurement._data_array, valid_data_array)
        assert measurement._data_array is not valid_data_array  # Ensure a copy was made
        assert measurement.unique_name == "test_measurement"
        assert measurement.display_name == "Test Measurement"

    def test_init_valid_data_array_no_pixel_coords(self, valid_data_array):
        # When
        data_array_no_pixel_coords = valid_data_array.copy(deep=True)
        del data_array_no_pixel_coords.coords['x']
        del data_array_no_pixel_coords.coords['y']
        # Then
        measurement = Measurement(data_array=data_array_no_pixel_coords)
        # Expect
        assert 'x' not in measurement._data_array.coords
        assert 'y' not in measurement._data_array.coords

    def test_init_valid_data_array_coordinate_edges(self, valid_data_array):
        # When
        data_array_with_edge_coords = valid_data_array.copy(deep=True)
        tof_bin_edges = sc.arange('t', 0, 11, 1, unit='s')
        x_bin_edges = sc.arange('x', 0, 6, 1, unit='m')
        y_bin_edges = sc.arange('y', 0, 6, 1, unit='m')
        data_array_with_edge_coords.coords['tof'] = tof_bin_edges
        data_array_with_edge_coords.coords['x'] = x_bin_edges
        data_array_with_edge_coords.coords['y'] = y_bin_edges
        # Then
        measurement = Measurement(data_array=data_array_with_edge_coords)
        # Expect
        assert sc.identical(
            measurement._data_array.coords['tof'], sc.midpoints(tof_bin_edges))
        assert sc.identical(
            measurement._data_array.coords['x'], sc.midpoints(x_bin_edges))
        assert sc.identical(
            measurement._data_array.coords['y'], sc.midpoints(y_bin_edges))

    def test_init_invalid_data_array_type(self):
        # When Then
        with pytest.raises(TypeError, match="data_array must be an instance of scipp.DataArray."):
            Measurement(data_array="not_a_data_array")

    @pytest.mark.parametrize("new_tof_coordinate, error, expected_message", [
        (None, ValueError, "data array must contain 'tof' coordinate for time-of-flight information."),
        (sc.scalar(5.0, unit='s'), ValueError, "data array must contain 'tof' coordinate for time-of-flight information."),
        (sc.arange('x', 0, 5, 1, unit='s'), ValueError, "'tof' coordinate must be of dimension 't'."),
        (sc.arange('t', 0, 10, 1, unit='m'), sc.UnitError, "'tof' coordinate must have a unit of time, such as \\('s'\\)."),  # noqa: E501
        (sc.arange('t', -5, 5, 1, unit='s'), ValueError, "time_of_flight values must be non-negative."),
    ], ids=[
        "missing_tof", 
        "scalar_tof",
        "wrong_dimension_tof",
        "invalid_unit_tof",
        "negative_tof",
    ])
    def test_init_invalid_tof_coordinates(self, valid_data_array, new_tof_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords['tof']
        if new_tof_coordinate is not None:
            invalid_data_array.coords['tof'] = new_tof_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize("new_x_coordinate, error, expected_message", [
        (sc.scalar(5.0, unit='m'), ValueError, "data array must contain 'x' coordinate for pixels."),
        (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'x' coordinate must be of dimension 'x'."),
        (sc.arange('x', 0, 5, 1, unit='s'), sc.UnitError, "'x' coordinate must have a unit of length, such as \\('m'\\)."),  # noqa: E501
    ], ids=[
        "scalar_x",
        "wrong_dimension_x",
        "invalid_unit_x",
    ])
    def test_init_invalid_x_coordinates(self, valid_data_array, new_x_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        invalid_data_array.coords['x'] = new_x_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize("new_y_coordinate, error, expected_message", [
        (sc.scalar(5.0, unit='m'), ValueError, "data array must contain 'y' coordinate for pixels."),
        (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'y' coordinate must be of dimension 'y'."),
        (sc.arange('y', 0, 5, 1, unit='s'), sc.UnitError, "'y' coordinate must have a unit of length, such as \\('m'\\)."),  # noqa: E501
        ], ids=[
            "scalar_y",
            "wrong_dimension_y",
            "invalid_unit_y",
        ])
    def test_init_invalid_y_coordinates(self, valid_data_array, new_y_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        invalid_data_array.coords['y'] = new_y_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize('dimension', ['x', 'y'], ids=['x_dimension', 'y_dimension'])
    def test_init_missing_coordinate_and_wrong_dimension(self, valid_data_array, dimension):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords[dimension]
        invalid_data_array = invalid_data_array.rename_dims({dimension: 'wrong_dim'})
        # Then Expect
        with pytest.raises(ValueError, match=f"data array must have an '{dimension}' dimension."):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize("path", ["small_scitiff.tiff", Path("small_scitiff.tiff")], ids=["str_path", "Path_object"])
    def test_from_scitiff_valid(self, path):
        # When Then
        measurement = Measurement.from_scitiff(
            filename=path,
            unique_name="test_measurement",
            display_name="Test Measurement"
        )
        # Expect
        assert measurement.unique_name == "test_measurement"
        assert measurement.display_name == "Test Measurement"
        assert 'tof' in measurement._data_array.coords
        assert 'x' in measurement._data_array.coords
        assert 'y' in measurement._data_array.coords

    def test_from_scitiff_invalid_path_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match="filename must be a string or Path object."):
            Measurement.from_scitiff(
                filename=12345,
            )

    def test_from_scitiff_non_existent_file(self):
        # When Then Expect
        with pytest.raises(RuntimeError, match="Failed to load SciTIFF file 'non_existent_file.tiff'"):
            Measurement.from_scitiff(
                filename="non_existent_file.tiff",
            )

    def test_from_scitiff_not_a_scitiff(self):
        # When
        path = 'small_tiff.tiff' # This is a regular TIFF, not a SciTIFF
        # Then Expect
        with pytest.raises(RuntimeError, match=f"Tiff file '{path}' not a proper SciTIFF file:"):
            Measurement.from_scitiff(
                filename=path,
            )

    @pytest.mark.parametrize("path", ["small_tiff.tiff", Path("small_tiff.tiff")], ids=["str_path", "Path_object"])
    def test_from_tiff_stack_valid_paths(self, path):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename=path,
            time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
            unique_name="test_measurement",
            display_name="Test Measurement"
        )
        # Expect
        assert measurement.unique_name == "test_measurement"
        assert measurement.display_name == "Test Measurement"
        assert 'tof' in measurement._data_array.coords
        assert 'x' not in measurement._data_array.coords
        assert 'y' not in measurement._data_array.coords

    @pytest.mark.parametrize("coord, expected", [
        (sc.arange('t', 0, 2400, 10, unit='us'), sc.arange('t', 0, 2400, 10, unit='us')), 
        (np.arange(0, 240, 1), sc.arange('t', 0, 240, 1, unit='s')), 
        ], ids=[
            "scipp_variable", 
            "numpy_array"
        ])
    def test_from_tiff_stack_valid_time_of_flights(self, coord, expected):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename="small_tiff.tiff",
            time_of_flight=coord,
        )
        # Expect
        assert sc.identical(measurement._data_array.coords['tof'], expected)

    @pytest.mark.parametrize("coord, expected", [
        (sc.arange('x', 0, 500, 10, unit='cm'), sc.arange('x', 0, 500, 10, unit='cm')),
        (np.arange(0, 50, 1), sc.arange('x', 0, 50, 1, unit='m')),
        ], ids=[
            "scipp_variable_x", 
            "numpy_array_x",
        ])
    def test_from_tiff_stack_valid_x_positions(self, coord, expected):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename="small_tiff.tiff",
            time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
            x_positions=coord,
        )
        # Expect
        assert 'x' in measurement._data_array.coords
        assert sc.identical(measurement._data_array.coords['x'], expected)

    @pytest.mark.parametrize("coord, expected", [
        (sc.arange('y', 0, 1000, 20, unit='cm'), sc.arange('y', 0, 1000, 20, unit='cm')),
        (np.arange(0, 100, 2), sc.arange('y', 0, 100, 2, unit='m')),
        ], ids=[
            "scipp_variable_y", 
            "numpy_array_y",
        ])
    def test_from_tiff_stack_valid_y_positions(self, coord, expected):

        # When Then
        measurement = Measurement.from_tiff_stack(
            filename="small_tiff.tiff",
            time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
            y_positions=coord,
        )
        # Expect
        assert 'y' in measurement._data_array.coords
        assert sc.identical(measurement._data_array.coords['y'], expected)

    @pytest.mark.parametrize("path, error", [(150, TypeError), ("non_existent_file.tiff", FileNotFoundError)], 
                             ids=["invalid_path_type", "non_existent_file"])
    def test_from_tiff_stack_invalid_path(self, path, error):
        # When Then Expect
        with pytest.raises(error):
            Measurement.from_tiff_stack(
                filename=path,
                time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
            )

    @pytest.mark.parametrize("coord, error, expected_message", [
        ("not_a_valid_type", TypeError, "time_of_flight must be a scipp Variable or a numpy Array."),
        (sc.arange('t', 0, 5, 1, unit='s'), ValueError, "Length of time_of_flight array does not match the number of frames in the TIFF stack."),  # noqa: E501
        (sc.arange('t', 0, 240, 1, unit='m'), sc.UnitError, "time_of_flight must have a unit of time, such as 's'"),
    ], ids=[
        "invalid_type",
        "wrong_length",
        "invalid_unit",
    ])
    def test_from_tiff_stack_invalid_time_of_flights(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename="small_tiff.tiff",
                time_of_flight=coord,
            )

    @pytest.mark.parametrize("coord, error, expected_message", [
        ("not_a_valid_type", TypeError, "x_positions must be a scipp Variable or a numpy Array."),
        (sc.arange('x', 0, 10, 1, unit='m'), ValueError, "Length of x_positions array does not match the number of pixels in the x dimension."),  # noqa: E501
        (sc.arange('x', 0, 50, 1, unit='s'), sc.UnitError, "x_positions must have a unit of length, such as 'm'"),
    ], ids=[
        "invalid_type_x",
        "wrong_length_x",
        "invalid_unit_x",
    ])
    def test_from_tiff_stack_invalid_x_positions(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename="small_tiff.tiff",
                time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
                x_positions=coord,
            )

    @pytest.mark.parametrize("coord, error, expected_message", [
        ("not_a_valid_type", TypeError, "y_positions must be a scipp Variable or a numpy Array."),
        (sc.arange('y', 0, 10, 1, unit='m'), ValueError, "Length of y_positions array does not match the number of pixels in the y dimension."),  # noqa: E501
        (sc.arange('y', 0, 50, 1, unit='s'), sc.UnitError, "y_positions must have a unit of length, such as 'm'"),
    ], ids=[
        "invalid_type_y",
        "wrong_length_y",
        "invalid_unit_y",
    ])
    def test_from_tiff_stack_invalid_y_positions(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename="small_tiff.tiff",
                time_of_flight=sc.arange('t', 0, 240, 1, unit='s'),
                y_positions=coord,
            )