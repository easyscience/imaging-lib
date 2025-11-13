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
        measurement = Measurement(name="TestMeasurement", data_array=valid_data_array)
        # Expect
        assert measurement.name == "TestMeasurement"
        assert sc.identical(measurement._data_array, valid_data_array)

    @pytest.mark.parametrize("new_tof_coordinate, error, expected_message", [
        (None, ValueError, "DataArray must contain 'tof' coordinate for time-of-flight information."),
        (sc.scalar(5.0, unit='s'), ValueError, "DataArray must contain 'tof' coordinate for time-of-flight information."),
        (sc.arange('x', 0, 5, 1, unit='s'), ValueError, "'tof' coordinate must be of dimension time: 't'."),
        (sc.arange('t', 0, 10, 1, unit='m'), sc.UnitError, "'tof' coordinate must have a unit of time, such as seconds \\('s'\\)."),  # noqa: E501
    ], ids=[
        "missing_tof", 
        "scalar_tof",
        "wrong_dimension_tof",
        "invalid_unit_tof",
    ])
    def test_init_invalid_tof_coordinates(self, valid_data_array, new_tof_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords['tof']
        if new_tof_coordinate is not None:
            invalid_data_array.coords['tof'] = new_tof_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(name="TestMeasurement", data_array=invalid_data_array)

    @pytest.mark.parametrize("new_x_coordinate, error, expected_message", [
        (None, ValueError, "DataArray must contain 'x' coordinate for pixels."),
        (sc.scalar(5.0, unit='m'), ValueError, "DataArray must contain 'x' coordinate for pixels."),
        (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'x' coordinate must be of dimension 'x'."),
        (sc.arange('x', 0, 5, 1, unit='s'), sc.UnitError, "'x' coordinate must have a unit of length, such as meters \\('m'\\)."),  # noqa: E501
    ], ids=[
        "missing_x", 
        "scalar_x",
        "wrong_dimension_x",
        "invalid_unit_x",
    ])
    def test_init_invalid_x_coordinates(self, valid_data_array, new_x_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords['x']
        if new_x_coordinate is not None:
            invalid_data_array.coords['x'] = new_x_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(name="TestMeasurement", data_array=invalid_data_array)

    @pytest.mark.parametrize("new_y_coordinate, error, expected_message", [
        (None, ValueError, "DataArray must contain 'y' coordinate for pixels."),
        (sc.scalar(5.0, unit='m'), ValueError, "DataArray must contain 'y' coordinate for pixels."),
        (sc.arange('t', 0, 10, 1, unit='m'), ValueError, "'y' coordinate must be of dimension 'y'."),
        (sc.arange('y', 0, 5, 1, unit='s'), sc.UnitError, "'y' coordinate must have a unit of length, such as meters \\('m'\\)."),  # noqa: E501
    ], ids=[
        "missing_y", 
        "scalar_y",
        "wrong_dimension_y",
        "invalid_unit_y",
    ])
    def test_init_invalid_y_coordinates(self, valid_data_array, new_y_coordinate, error, expected_message):
        # When
        invalid_data_array = valid_data_array.copy(deep=True)
        del invalid_data_array.coords['y']
        if new_y_coordinate is not None:
            invalid_data_array.coords['y'] = new_y_coordinate
        # Then Expect
        with pytest.raises(error, match=expected_message):
            Measurement(name="TestMeasurement", data_array=invalid_data_array)
