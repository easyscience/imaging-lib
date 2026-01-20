from pathlib import Path

import numpy as np
import pytest
import scipp as sc

from easyimaging import Measurement


class TestMeasurement:
    @pytest.fixture
    def valid_data_array(self):
        tof = sc.arange('t', 0, 10, 1, unit='s')
        x = sc.arange('x', 0, 7, 1, unit='m')
        y = sc.arange('y', 0, 7, 1, unit='m')
        data = sc.zeros(dims=('x', 'y', 't'), shape=(6, 6, 10))
        return sc.DataArray(data=data, coords={'tof': tof, 'x': x, 'y': y})

    def test_init_valid_data_array(self, valid_data_array):
        # When Then
        measurement = Measurement(data_array=valid_data_array, unique_name='test_measurement', display_name='Test Measurement')
        # Expect
        assert measurement._data_array is not valid_data_array  # Ensure a copy was made
        assert sc.any(measurement._data_array.masks['non_finite']).value is False
        del measurement._data_array.coords['x_pixels']
        del measurement._data_array.coords['y_pixels']
        del measurement._data_array.masks['non_finite']
        assert sc.identical(measurement._data_array, valid_data_array)
        assert measurement.unique_name == 'test_measurement'
        assert measurement.display_name == 'Test Measurement'
        assert not hasattr(measurement, '_rebinned_data_array')

    def test_init_valid_data_array_no_xy_coords(self, valid_data_array):
        # When
        data_array_no_xy_coords = valid_data_array.copy(deep=True)
        del data_array_no_xy_coords.coords['x']
        del data_array_no_xy_coords.coords['y']
        # Then
        measurement = Measurement(data_array=data_array_no_xy_coords)
        # Expect
        assert 'x' not in measurement._data_array.coords
        assert 'y' not in measurement._data_array.coords
        assert 'x_pixels' in measurement._data_array.coords
        assert measurement._data_array.coords.is_edges('x_pixels')
        assert 'y_pixels' in measurement._data_array.coords
        assert measurement._data_array.coords.is_edges('y_pixels')

    def test_init_valid_data_array_coordinate_not_edges(self, valid_data_array):
        # When
        data_array_with_center_coords = valid_data_array.copy(deep=True)
        x_bin_edges = sc.arange('x', 0, 6, 1, unit='m')
        y_bin_edges = sc.arange('y', 0, 6, 1, unit='m')
        data_array_with_center_coords.coords['x'] = x_bin_edges
        data_array_with_center_coords.coords['y'] = y_bin_edges
        # Then
        measurement = Measurement(data_array=data_array_with_center_coords)
        # Expect
        assert sc.identical(measurement._data_array.coords['x'], sc.arange('x', -0.5, 6.5, 1, unit='m'))
        assert sc.identical(measurement._data_array.coords['y'], sc.arange('y', -0.5, 6.5, 1, unit='m'))

    @pytest.mark.parametrize('value', [np.nan, np.inf], ids=['nan', 'inf'])
    def test_init_valid_data_array_with_nonfinite_values(self, valid_data_array, value):
        # When
        data_array_with_nonfinite = valid_data_array.copy(deep=True)
        data_array_with_nonfinite.data['x', 0]['y', 0]['t', 0] = value
        # Then
        measurement = Measurement(data_array=data_array_with_nonfinite)
        # Expect
        assert sc.any(measurement._data_array.masks['non_finite']).value is True
        assert measurement._data_array.masks['non_finite']['x', 0]['y', 0]['t', 0].value is True

    def test_init_invalid_data_array_type(self):
        # When Then
        with pytest.raises(TypeError, match='data_array must be an instance of scipp.DataArray.'):
            Measurement(data_array='not_a_data_array')

    @pytest.mark.parametrize(
        'new_tof_coordinate, error, expected_message',
        [
            (None, ValueError, "data array must contain 'tof' coordinate for time-of-flight information."),
            (sc.scalar(5.0, unit='s'), ValueError, "data array must contain 'tof' coordinate for time-of-flight information."),
            (sc.arange('x', 0, 6, 1, unit='s'), ValueError, "'tof' coordinate must be of dimension 't'."),
            (
                sc.arange('t', 0, 10, 1, unit='m'),
                sc.UnitError,
                "'tof' coordinate must have a unit of time, such as \\('s'\\).",
            ),  # noqa: E501 # fmt: skip
            (sc.arange('t', -5, 5, 1, unit='s'), ValueError, 'time_of_flight values must be non-negative.'),
        ],
        ids=[
            'missing_tof',
            'scalar_tof',
            'wrong_dimension_tof',
            'invalid_unit_tof',
            'negative_tof',
        ],
    )
    def test_init_invalid_tof_coordinates(self, valid_data_array, new_tof_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords['tof']
        if new_tof_coordinate is not None:
            invalid_data_array.coords['tof'] = new_tof_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize(
        'new_x_coordinate, error, expected_message',
        [
            (sc.scalar(5.0, unit='m'), ValueError, "data array must contain 'x' coordinate for pixels."),
            (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'x' coordinate must be of dimension 'x'."),
            (sc.arange('x', 0, 6, 1, unit='s'), sc.UnitError, "'x' coordinate must have a unit of length, such as \\('m'\\)."),  # noqa: E501
        ],
        ids=[
            'scalar_x',
            'wrong_dimension_x',
            'invalid_unit_x',
        ],
    )
    def test_init_invalid_x_coordinates(self, valid_data_array, new_x_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        invalid_data_array.coords['x'] = new_x_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(data_array=invalid_data_array)

    @pytest.mark.parametrize(
        'new_y_coordinate, error, expected_message',
        [
            (sc.scalar(5.0, unit='m'), ValueError, "data array must contain 'y' coordinate for pixels."),
            (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'y' coordinate must be of dimension 'y'."),
            (sc.arange('y', 0, 6, 1, unit='s'), sc.UnitError, "'y' coordinate must have a unit of length, such as \\('m'\\)."),  # noqa: E501
        ],
        ids=[
            'scalar_y',
            'wrong_dimension_y',
            'invalid_unit_y',
        ],
    )
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

    @pytest.mark.parametrize('path', ['small_scitiff.tiff', Path('small_scitiff.tiff')], ids=['str_path', 'Path_object'])
    def test_from_scitiff_valid(self, path):
        # When Then
        measurement = Measurement.from_scitiff(filename=path, unique_name='test_measurement', display_name='Test Measurement')
        # Expect
        assert measurement.unique_name == 'test_measurement'
        assert measurement.display_name == 'Test Measurement'
        assert 'tof' in measurement._data_array.coords
        assert 'x' in measurement._data_array.coords
        assert 'y' in measurement._data_array.coords
        assert 'non_finite' in measurement._data_array.masks
        assert sc.any(measurement._data_array.masks['non_finite']).value is True

    def test_from_scitiff_invalid_path_type(self):
        # When Then Expect
        with pytest.raises(TypeError, match='filename must be a string or Path object.'):
            Measurement.from_scitiff(
                filename=12345,
            )

    def test_from_scitiff_non_existent_file(self):
        # When Then Expect
        with pytest.raises(RuntimeError, match="Failed to load SciTIFF file 'non_existent_file.tiff'"):
            Measurement.from_scitiff(
                filename='non_existent_file.tiff',
            )

    def test_from_scitiff_not_a_scitiff(self):
        # When
        path = 'small_tiff.tiff'  # This is a regular TIFF, not a SciTIFF
        # Then Expect
        with pytest.raises(RuntimeError, match=f"Tiff file '{path}' not a proper SciTIFF file:"):
            Measurement.from_scitiff(
                filename=path,
            )

    @pytest.mark.parametrize('path', ['small_tiff.tiff', Path('small_tiff.tiff')], ids=['str_path', 'Path_object'])
    def test_from_tiff_stack_valid_paths(self, path):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename=path,
            time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
            unique_name='test_measurement',
            display_name='Test Measurement',
        )
        # Expect
        assert measurement.unique_name == 'test_measurement'
        assert measurement.display_name == 'Test Measurement'
        assert 'tof' in measurement._data_array.coords
        assert 'x' not in measurement._data_array.coords
        assert 'y' not in measurement._data_array.coords

    @pytest.mark.parametrize(
        'coord, expected',
        [
            (sc.arange('t', 0, 2400, 10, unit='us'), sc.arange('t', 0, 2400, 10, unit='us')),
            (np.arange(0, 240, 1), sc.arange('t', 0, 240, 1, unit='s')),
        ],
        ids=['scipp_variable', 'numpy_array'],
    )
    def test_from_tiff_stack_valid_time_of_flights(self, coord, expected):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename='small_tiff.tiff',
            time_of_flights=coord,
        )
        # Expect
        assert sc.identical(measurement._data_array.coords['tof'], expected)

    @pytest.mark.parametrize(
        'coord, expected',
        [
            (sc.arange('x', 0, 510, 10, unit='cm'), sc.arange('x', 0, 510, 10, unit='cm')),
            (np.arange(0, 51, 1), sc.arange('x', 0, 51, 1, unit='m')),
        ],
        ids=[
            'scipp_variable_x',
            'numpy_array_x',
        ],
    )
    def test_from_tiff_stack_valid_x_positions(self, coord, expected):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename='small_tiff.tiff',
            time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
            x_positions=coord,
        )
        # Expect
        assert 'x' in measurement._data_array.coords
        assert sc.identical(measurement._data_array.coords['x'], expected)

    @pytest.mark.parametrize(
        'coord, expected',
        [
            (sc.arange('y', 0, 1020, 20, unit='cm'), sc.arange('y', 0, 1020, 20, unit='cm')),
            (np.arange(0, 102, 2), sc.arange('y', 0, 102, 2, unit='m')),
        ],
        ids=[
            'scipp_variable_y',
            'numpy_array_y',
        ],
    )
    def test_from_tiff_stack_valid_y_positions(self, coord, expected):
        # When Then
        measurement = Measurement.from_tiff_stack(
            filename='small_tiff.tiff',
            time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
            y_positions=coord,
        )
        # Expect
        assert 'y' in measurement._data_array.coords
        assert sc.identical(measurement._data_array.coords['y'], expected)

    @pytest.mark.parametrize(
        'path, error',
        [(150, TypeError), ('non_existent_file.tiff', FileNotFoundError)],
        ids=['invalid_path_type', 'non_existent_file'],
    )
    def test_from_tiff_stack_invalid_path(self, path, error):
        # When Then Expect
        with pytest.raises(error):
            Measurement.from_tiff_stack(
                filename=path,
                time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
            )

    @pytest.mark.parametrize(
        'coord, error, expected_message',
        [
            ('not_a_valid_type', TypeError, 'time_of_flight must be a scipp Variable or a numpy Array.'),
            (
                sc.arange('t', 0, 5, 1, unit='s'),
                ValueError,
                'Length of time_of_flight array does not match the number of frames in the TIFF stack.',
            ),  # noqa: E501
            (sc.arange('t', 0, 240, 1, unit='m'), sc.UnitError, "time_of_flight must have a unit of time, such as 's'"),
        ],
        ids=[
            'invalid_type',
            'wrong_length',
            'invalid_unit',
        ],
    )
    def test_from_tiff_stack_invalid_time_of_flights(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename='small_tiff.tiff',
                time_of_flights=coord,
            )

    @pytest.mark.parametrize(
        'coord, error, expected_message',
        [
            ('not_a_valid_type', TypeError, 'x_positions must be a scipp Variable or a numpy Array.'),
            (
                sc.arange('x', 0, 10, 1, unit='m'),
                ValueError,
                'Length of x_positions array does not match the number of pixels in the x dimension.',
            ),  # noqa: E501
            (sc.arange('x', 0, 50, 1, unit='s'), sc.UnitError, "x_positions must have a unit of length, such as 'm'"),
        ],
        ids=[
            'invalid_type_x',
            'wrong_length_x',
            'invalid_unit_x',
        ],
    )
    def test_from_tiff_stack_invalid_x_positions(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename='small_tiff.tiff',
                time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
                x_positions=coord,
            )

    @pytest.mark.parametrize(
        'coord, error, expected_message',
        [
            ('not_a_valid_type', TypeError, 'y_positions must be a scipp Variable or a numpy Array.'),
            (
                sc.arange('y', 0, 10, 1, unit='m'),
                ValueError,
                'Length of y_positions array does not match the number of pixels in the y dimension.',
            ),  # noqa: E501
            (sc.arange('y', 0, 50, 1, unit='s'), sc.UnitError, "y_positions must have a unit of length, such as 'm'"),
        ],
        ids=[
            'invalid_type_y',
            'wrong_length_y',
            'invalid_unit_y',
        ],
    )
    def test_from_tiff_stack_invalid_y_positions(self, coord, error, expected_message):
        # When Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement.from_tiff_stack(
                filename='small_tiff.tiff',
                time_of_flights=sc.arange('t', 0, 240, 1, unit='s'),
                y_positions=coord,
            )

    def test_rebin_full(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then
        measurement.rebin(dimensions={'x': 2, 'y': 3})
        # Expect
        assert measurement._data_array.sizes['x'] == 3
        assert measurement._data_array.sizes['y'] == 2
        assert measurement._data_array.sizes['t'] == 10
        assert sc.identical(measurement._data_array.coords['x_pixels'], sc.arange('x', 0, 7, 2))
        assert sc.identical(measurement._data_array.coords['y_pixels'], sc.arange('y', 0, 7, 3))
        assert sc.identical(measurement._data_array.coords['x'], sc.arange('x', 0, 7, 2, unit='m'))
        assert sc.identical(measurement._data_array.coords['y'], sc.arange('y', 0, 7, 3, unit='m'))
        assert sc.identical(measurement._data_array, measurement._rebinned_data_array)
        assert not sc.identical(measurement._data_array, measurement._full_data_array)
        assert 'non_finite' in measurement._data_array.masks

    def test_rebin_partial(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then
        measurement.rebin(dimensions={'x': 3})
        # Expect
        assert measurement._data_array.sizes['x'] == 2
        assert measurement._data_array.sizes['y'] == 6
        assert sc.identical(measurement._data_array.coords['x_pixels'], sc.arange('x', 0, 7, 3))
        assert sc.identical(measurement._data_array.coords['y_pixels'], sc.arange('y', 0, 7, 1))
        assert sc.identical(measurement._data_array.coords['x'], sc.arange('x', 0, 7, 3, unit='m'))
        assert sc.identical(measurement._data_array.coords['y'], sc.arange('y', 0, 7, 1, unit='m'))

    @pytest.mark.parametrize('values, result', 
                             [([np.nan, np.nan], True), 
                              ([np.nan, 5.0], False), 
                              ([np.inf, np.inf], True), 
                              ([np.inf, 10.0], False), 
                              ([np.nan, np.inf], True)], 
                             ids=['all_nan', 'nan_and_finite', 'all_inf', 'inf_and_finite', 'nan_and_inf'])
    def test_rebin_with_masked_data(self, valid_data_array, values, result):
        # When
        data_array_with_masked = valid_data_array.copy(deep=True)
        data_array_with_masked.data['x', 0]['y', 0:2]['t', 0] = values
        measurement = Measurement(data_array=data_array_with_masked)
        # Then
        measurement.rebin(dimensions={'y': 2})
        # Expect
        assert measurement._data_array.sizes['y'] == 3
        assert sc.any(measurement._data_array.masks['non_finite']).value is result

    def test_rebin_no_xy_coords(self, valid_data_array):
        # When
        data_array_no_xy_coords = valid_data_array.copy(deep=True)
        del data_array_no_xy_coords.coords['x']
        del data_array_no_xy_coords.coords['y']
        measurement = Measurement(data_array=data_array_no_xy_coords)
        # Then
        measurement.rebin(dimensions={'x': 2, 'y': 3})
        # Expect
        assert measurement._data_array.sizes['x'] == 3
        assert measurement._data_array.sizes['y'] == 2
        assert sc.identical(measurement._data_array.coords['x_pixels'], sc.arange('x', 0, 7, 2))
        assert sc.identical(measurement._data_array.coords['y_pixels'], sc.arange('y', 0, 7, 3))
        assert 'x' not in measurement._data_array.coords
        assert 'y' not in measurement._data_array.coords

    def test_rebin_twice(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        measurement.rebin(dimensions={'x': 2, 'y': 3})
        # Then
        measurement.rebin(dimensions={'x': 3})
        # Expect
        assert measurement._data_array.sizes['x'] == 1
        assert measurement._data_array.sizes['y'] == 2
        assert sc.identical(measurement._data_array.coords['x_pixels'], sc.arange('x', 0, 7, 6))
        assert sc.identical(measurement._data_array.coords['y_pixels'], sc.arange('y', 0, 7, 3))

    def test_rebin_unity(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        data_array = measurement._data_array
        # Then
        measurement.rebin(dimensions={'x': 1, 'y': 1})
        # Expect
        assert not hasattr(measurement, '_rebinned_data_array')
        assert sc.identical(measurement._data_array, data_array)

    def test_rebin_unity_second_time(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        measurement.rebin(dimensions={'x': 2, 'y': 3})
        data_array = measurement._data_array
        # Then
        measurement.rebin(dimensions={'x': 1, 'y': 1})
        # Expect
        assert hasattr(measurement, '_rebinned_data_array')
        assert sc.identical(measurement._data_array, data_array)

    def test_rebin_float_integers(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then
        measurement.rebin(dimensions={'x': 2.0, 'y': 3.0})
        # Expect
        assert measurement._data_array.sizes['x'] == 3
        assert measurement._data_array.sizes['y'] == 2
        assert sc.identical(measurement._data_array.coords['x_pixels'], sc.arange('x', 0, 7, 2))
        assert sc.identical(measurement._data_array.coords['y_pixels'], sc.arange('y', 0, 7, 3))
        assert sc.identical(measurement._data_array.coords['x'], sc.arange('x', 0, 7, 2, unit='m'))
        assert sc.identical(measurement._data_array.coords['y'], sc.arange('y', 0, 7, 3, unit='m'))

    def test_rebin_invalid_dimension_type(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(TypeError, match='dimensions must be a dictionary mapping dimension names to rebin factors.'):
            measurement.rebin(dimensions=['x'])

    def test_rebin_time_dimension(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(ValueError, match='Rebinning of the time-of-flight'):
            measurement.rebin(dimensions={'t': 2, 'x': 2})

    def test_rebin_invalid_dimensions_key_type(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(TypeError, match="Dimension keys must be strings. Got <class 'int'> for 0 instead."):
            measurement.rebin(dimensions={0: 2})

    def test_rebin_invalid_dimension_name(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(KeyError, match="Dimension 'z' not a valid dimension for rebinning. Should be one of"):
            measurement.rebin(dimensions={'z': 2})

    def test_rebin_invalid_dimensions_value_type(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(ValueError, match="Rebin size for dimension 'x' must be a positive integer of at least 1."):
            measurement.rebin(dimensions={'x': 'not_an_integer'})

    def test_rebin_invalid_dimensions_value_zero(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(ValueError, match="Rebin size for dimension 'x' must be a positive integer of at least 1."):
            measurement.rebin(dimensions={'x': 0})

    def test_rebin_invalid_dimensions_value_negative(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(ValueError, match="Rebin size for dimension 'x' must be a positive integer of at least 1."):
            measurement.rebin(dimensions={'x': -2})

    def test_rebin_invalid_dimensions_value_non_divisable(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(ValueError, match="Dimension 'x' with size 6 is not evenly divisible by rebin size 4."):
            measurement.rebin(dimensions={'x': 4})

    def test_revert_rebin(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        measurement.rebin(dimensions={'x': 2, 'y': 3})
        # Then
        measurement.revert_rebin()
        # Expect
        assert not hasattr(measurement, '_rebinned_data_array')
        del measurement._data_array.coords['x_pixels']
        del measurement._data_array.coords['y_pixels']
        del measurement._data_array.masks['non_finite']
        assert sc.identical(measurement._data_array, valid_data_array)

    #Without making image comparisons, this is the best we can do to test the plot function
    @pytest.mark.parametrize('time_of_flight', [None, 0, sc.scalar(5.0, unit='s')], ids=['sum', 'indice', 'scipp_scalar'])
    def test_plot(self, valid_data_array, time_of_flight):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        measurement.plot(time_of_flight=time_of_flight)  # Just ensure no exception is raised

    def test_plot_invalid_time_of_flight_type(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(TypeError, match='time_of_flight must be an integer, scipp Variable, or None.'):
            measurement.plot(time_of_flight='not_a_valid_type')

    def test_plot_invalid_time_of_flight_unit(self, valid_data_array):
        # When
        measurement = Measurement(data_array=valid_data_array)
        # Then Expect
        with pytest.raises(sc.UnitError, match="time_of_flight variable must have a unit of time such as 's'"):
            measurement.plot(time_of_flight=sc.scalar(5.0, unit='m'))