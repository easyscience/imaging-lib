import pytest
import scipp as sc
from scipp import UnitError

from easyimaging.measurement.regions import RectROI


class TestRectROI:
    @pytest.fixture
    def roi_basic(self):
        return RectROI((10, 50), [20, 80])

    @pytest.fixture
    def roi_with_physical_coords(self):
        x_start = sc.scalar(0.0, unit='m')
        x_end = sc.scalar(10.0, unit='m')
        y_start = sc.scalar(0.0, unit='m')
        y_end = sc.scalar(5.0, unit='m')
        return RectROI((10, 50), (20, 80), (x_start, x_end), (y_start, y_end), unique_name='test_roi', display_name='Test ROI')

    def test_init_valid_physical_coords(self, roi_with_physical_coords):
        # When Then
        roi = roi_with_physical_coords
        # Expect
        assert roi.unique_name == 'test_roi'
        assert roi.display_name == 'Test ROI'
        assert roi.x_pixel_start == 10
        assert roi.x_pixel_end == 50
        assert roi.y_pixel_start == 20
        assert roi.y_pixel_end == 80
        assert roi._has_physical_coords is True
        assert sc.identical(roi.x_start, sc.scalar(0.0, unit='m'))
        assert sc.identical(roi.x_end, sc.scalar(10.0, unit='m'))
        assert sc.identical(roi.y_start, sc.scalar(0.0, unit='m'))
        assert sc.identical(roi.y_end, sc.scalar(5.0, unit='m'))

    def test_init_valid_pixel_only(self, roi_basic):
        # When Then
        roi = roi_basic
        # Expect
        assert roi.unique_name.startswith('RectROI_')
        assert roi.display_name.startswith('RectROI_')
        assert roi.x_pixel_start == 10
        assert roi.x_pixel_end == 50
        assert roi.y_pixel_start == 20
        assert roi.y_pixel_end == 80
        assert roi._has_physical_coords is False

    def test_init_invalid_single_physical_coords(self):
        # When
        x_start = sc.scalar(0.0, unit='m')
        x_end = sc.scalar(10.0, unit='m')
        # Then Expect
        with pytest.raises(ValueError, match='Both x_range and y_range must be provided together or not at all.'):
            RectROI((10, 50), (20, 80), (x_start, x_end))

    def test_init_invalid_negative_pixel_indices(self):
        # When Then Expect
        with pytest.raises(ValueError, match='Pixel indices must be non-negative integers.'):
            RectROI((-10, 50), (20, 80))

    @pytest.mark.parametrize(
        'x_pixel_range',
        [
            (10,),  # Only one element
            (10, 20, 30),  # Three elements
            {'test': 4},  # Not a sequence
            (10, 2.0),  # Second element not an int
        ],
        ids=['one_element', 'three_elements', 'not_a_sequence', 'second_not_int'],
    )
    def test_init_invalid_x_pixel_range_arguments(self, x_pixel_range):
        # When Then Expect
        with pytest.raises(TypeError, match='x_pixel_range must be a tuple or a list of two integers'):
            RectROI(x_pixel_range, (20, 80))

    @pytest.mark.parametrize(
        'x_range, error, message',
        [
            (
                (sc.scalar(0.0, unit='m'),),
                TypeError,
                'x_range must be a tuple or a list of two scipp scalars',
            ),  # Only one element
            (
                (sc.scalar(0.0, unit='m'), sc.scalar(1.0, unit='m'), sc.scalar(2.0, unit='m')),
                TypeError,
                'x_range must be a tuple or a list of two scipp scalars',
            ),  # Three elements  # noqa: E501
            (
                {'test': sc.scalar(1.0, unit='m')},
                TypeError,
                'x_range must be a tuple or a list of two scipp scalars',
            ),  # Not a sequence  # noqa: E501
            (
                (sc.scalar(0.0, unit='m'), 2.0),
                TypeError,
                'x_range must be a tuple or a list of two scipp scalars',
            ),  # Second element not a scipp Variable  # noqa: E501
            (
                (sc.scalar(0.0, unit='m'), sc.array(dims=['x'], values=[1.0, 2.0], unit='m')),
                ValueError,
                'Physical coordinates must be a scipp scalar \(0-dimensional Variable\).',
            ),  # Second element not a scalar  # noqa: E501
            (
                (sc.scalar(0.0, unit='m'), sc.scalar(1.0, unit='s')),
                UnitError,
                "Physical coordinates must be a scipp scalar with a unit of length \(e.g., 'm'\).",
            ),  # Second element with wrong unit  # noqa: E501
        ],
        ids=[
            'one_element',
            'three_elements',
            'not_a_sequence',
            'second_not_variable',
            'second_not_scalar',
            'second_wrong_unit',
        ],
    )
    def test_init_invalid_x_range_arguments(self, x_range, error, message):
        # When Then Expect
        with pytest.raises(error, match=message):
            RectROI((10, 50), (20, 80), x_range, (sc.scalar(0.0, unit='m'), sc.scalar(1.0, unit='m')))

    def test_set_physical_coord_range(self, roi_basic):
        # When
        roi = roi_basic
        x_start = sc.scalar(0.0, unit='m')
        x_end = sc.scalar(10.0, unit='m')
        y_start = sc.scalar(0.0, unit='m')
        y_end = sc.scalar(5.0, unit='m')
        # Then
        roi.set_physical_coord_range((x_start, x_end), (y_start, y_end))
        # Expect
        assert roi._has_physical_coords is True
        assert sc.identical(roi.x_start, x_start)
        assert sc.identical(roi.x_end, x_end)
        assert sc.identical(roi.y_start, y_start)
        assert sc.identical(roi.y_end, y_end)

    def test_delete_physical_coord_range(self, roi_with_physical_coords):
        # When
        roi = roi_with_physical_coords
        # Then
        roi.delete_physical_coord_range()
        # Expect
        assert roi._has_physical_coords is False
        assert not hasattr(roi, '_x_start')
        assert not hasattr(roi, '_x_end')
        assert not hasattr(roi, '_y_start')
        assert not hasattr(roi, '_y_end')

    def test_delete_physical_coord_range_not_set_raises(self, roi_basic):
        # When
        roi = roi_basic
        # Then Expect
        with pytest.raises(ValueError, match='Cannot delete physical coordinate ranges because they are not set.'):
            roi.delete_physical_coord_range()

    @pytest.mark.parametrize(
        'attribute, value',
        [('x_pixel_start', 10), ('x_pixel_end', 50), ('y_pixel_start', 20), ('y_pixel_end', 80)],
        ids=['x_pixel_start', 'x_pixel_end', 'y_pixel_start', 'y_pixel_end'],
    )
    def test_pixel_coordinate_getter(self, roi_basic, attribute, value):
        # When
        roi = roi_basic
        # Then Expect
        assert getattr(roi, attribute) == value

    @pytest.mark.parametrize(
        'attribute',
        ['x_pixel_start', 'x_pixel_end', 'y_pixel_start', 'y_pixel_end'],
        ids=['x_pixel_start', 'x_pixel_end', 'y_pixel_start', 'y_pixel_end'],
    )
    def test_pixel_coordinate_setter(self, roi_basic, attribute):
        # When
        roi = roi_basic
        # Then
        setattr(roi, attribute, 15)
        # Expect
        assert getattr(roi, attribute) == 15

    @pytest.mark.parametrize(
        'attribute', ['x_pixel_end', 'y_pixel_start', 'y_pixel_end'], ids=['x_pixel_end', 'y_pixel_start', 'y_pixel_end']
    )
    @pytest.mark.parametrize(
        'invalid_value, error, message',
        [
            (-5, ValueError, 'index must be non-negative'),  # Negative integer
            (3.5, TypeError, 'index must be an integer'),  # Float
            ('10', TypeError, 'index must be an integer'),  # String
        ],
        ids=['negative_integer', 'float', 'string'],
    )
    def test_pixel_coordinate_setter_invalid(self, roi_basic, invalid_value, error, message, attribute):
        # When
        roi = roi_basic
        # Then Expect
        with pytest.raises(error, match=message):
            setattr(roi, attribute, invalid_value)

    @pytest.mark.parametrize(
        'attribute',
        [
            'x_start',
            'x_end',
            'y_start',
            'y_end',
        ],
    )
    def test_physical_coordinate_getter_not_set_raises(self, roi_basic, attribute):
        # When
        roi = roi_basic
        # Then Expect
        with pytest.raises(ValueError, match='Physical coordinate ranges are not set for this ROI.'):
            getattr(roi, attribute)

    @pytest.mark.parametrize(
        'attribute, scalar',
        [
            ('x_start', sc.scalar(0.0, unit='m')),
            ('x_end', sc.scalar(10.0, unit='m')),
            ('y_start', sc.scalar(0.0, unit='m')),
            ('y_end', sc.scalar(5.0, unit='m')),
        ],
        ids=['x_start', 'x_end', 'y_start', 'y_end'],
    )
    def test_physical_coordinate_getter(self, roi_with_physical_coords, attribute, scalar):
        # When
        roi = roi_with_physical_coords
        # Then Expect
        assert sc.identical(getattr(roi, attribute), scalar)
        assert getattr(roi, attribute) is not getattr(roi, f'_{attribute}')  # Ensure a copy is returned

    @pytest.mark.parametrize(
        'attribute', ['x_start', 'x_end', 'y_start', 'y_end'], ids=['x_start', 'x_end', 'y_start', 'y_end']
    )
    def test_physical_coordinate_setter(self, roi_with_physical_coords, attribute):
        # When
        roi = roi_with_physical_coords
        # Then
        new_value = sc.scalar(1.0, unit='m')
        setattr(roi, attribute, new_value)
        # Expect
        assert sc.identical(getattr(roi, attribute), new_value)

    @pytest.mark.parametrize(
        'attribute', ['x_start', 'x_end', 'y_start', 'y_end'], ids=['x_start', 'x_end', 'y_start', 'y_end']
    )
    def test_physical_coordinate_setter_setting_single_coordinate(self, roi_basic, attribute):
        # When
        roi = roi_basic
        # Then Expect
        with pytest.raises(ValueError, match=f'Cannot set {attribute} before setting all physical coordinate ranges.'):
            setattr(roi, attribute, sc.scalar(1.0, unit='m'))

    @pytest.mark.parametrize(
        'attribute', ['x_start', 'x_end', 'y_start', 'y_end'], ids=['x_start', 'x_end', 'y_start', 'y_end']
    )
    @pytest.mark.parametrize(
        'invalid_value, error, message',
        [
            (10.0, TypeError, 'must be a scipp scalar.'),  # Not a scipp Variable
            (
                sc.array(dims=['x'], values=[1.0, 2.0], unit='m'),
                ValueError,
                'must be a scipp scalar \(0-dimensional Variable\).',
            ),  # Not a scalar  # noqa: E501
            (
                sc.scalar(1.0, unit='s'),
                UnitError,
                "must be a scipp scalar with a unit of length \(e.g., 'm'\).",
            ),  # Wrong unit  # noqa: E501
        ],
        ids=['not_scipp_variable', 'not_scalar', 'wrong_unit'],
    )
    def test_physical_coordinate_setter_invalid(self, attribute, roi_with_physical_coords, invalid_value, error, message):
        # When
        roi = roi_with_physical_coords
        # Then Expect
        with pytest.raises(error, match=message):
            setattr(roi, attribute, invalid_value)

    def test_pixel_slice(self, roi_basic):
        # When
        roi = roi_basic
        # Then
        pixel_slice = roi.pixel_slice()
        # Expect
        assert pixel_slice[0] == slice(sc.scalar(10), sc.scalar(50))
        assert pixel_slice[1] == slice(sc.scalar(20), sc.scalar(80))

    def test_slice_valid(self, roi_with_physical_coords):
        # When
        roi = roi_with_physical_coords
        # Then
        physical_slice = roi.slice()
        # Expect
        assert physical_slice[0] == slice(sc.scalar(0.0, unit='m'), sc.scalar(10.0, unit='m'))
        assert physical_slice[1] == slice(sc.scalar(0.0, unit='m'), sc.scalar(5.0, unit='m'))

    def test_slice_no_physical_coords_set(self, roi_basic):
        # When
        roi = roi_basic
        # Then Expect
        with pytest.raises(ValueError, match='Physical coordinate ranges are not set for this ROI.'):
            roi.slice()

    def test_to_dict_only_pixel_ranges(self, roi_basic):
        # When Then
        roi_dict = roi_basic.to_dict()
        # Expect
        assert roi_dict['x_pixel_range'] == [roi_basic.x_pixel_start, roi_basic.x_pixel_end]
        assert roi_dict['y_pixel_range'] == [roi_basic.y_pixel_start, roi_basic.y_pixel_end]

    def test_to_dict_with_physical_coords(self, roi_with_physical_coords):
        # When Then
        roi_dict = roi_with_physical_coords.to_dict()
        # Expect
        assert roi_dict['x_pixel_range'] == [roi_with_physical_coords.x_pixel_start, roi_with_physical_coords.x_pixel_end]
        assert roi_dict['y_pixel_range'] == [roi_with_physical_coords.y_pixel_start, roi_with_physical_coords.y_pixel_end]
        assert isinstance(roi_dict['x_range'], list) and len(roi_dict['x_range']) == 2
        assert isinstance(roi_dict['y_range'], list) and len(roi_dict['y_range']) == 2
        assert isinstance(roi_dict['x_range'][0], dict) and roi_dict['x_range'][0]['@module'] == 'scipp'
        assert isinstance(roi_dict['y_range'][0], dict) and roi_dict['y_range'][0]['@module'] == 'scipp'
        assert roi_dict['display_name'] == roi_with_physical_coords.display_name
        assert roi_dict['unique_name'] == roi_with_physical_coords.unique_name

    def test_to_dict_skip_parameter(self, roi_with_physical_coords):
        # When Then
        roi_dict = roi_with_physical_coords.to_dict(skip='unique_name')
        # Expect
        assert 'unique_name' not in roi_dict

    def test_from_dict_only_pixel_ranges(self):
        # Given
        input_dict = {
            '@module': 'easyimaging.measurement.regions',
            '@class': 'RectROI',
            'x_pixel_range': [10, 50],
            'y_pixel_range': [20, 80],
        }
        # When Then
        roi = RectROI.from_dict(input_dict)
        # Expect
        assert roi.x_pixel_start == 10
        assert roi.x_pixel_end == 50
        assert roi.y_pixel_start == 20
        assert roi.y_pixel_end == 80
        assert roi._has_physical_coords is False

    def test_from_dict_with_physical_coords(self, roi_with_physical_coords):
        # Given
        roi = roi_with_physical_coords
        input_dict = roi.to_dict(skip='unique_name')  # Skip unique_name to allow auto-generation in from_dict
        # When Then
        roi_from_dict = RectROI.from_dict(input_dict)
        # Expect
        assert roi_from_dict.x_pixel_start == roi.x_pixel_start
        assert roi_from_dict.x_pixel_end == roi.x_pixel_end
        assert roi_from_dict.y_pixel_start == roi.y_pixel_start
        assert roi_from_dict.y_pixel_end == roi.y_pixel_end
        assert roi_from_dict._has_physical_coords is True
        assert sc.identical(roi_from_dict.x_start, roi.x_start)
        assert sc.identical(roi_from_dict.x_end, roi.x_end)
        assert sc.identical(roi_from_dict.y_start, roi.y_start)
        assert sc.identical(roi_from_dict.y_end, roi.y_end)
        assert 'unique_name' not in input_dict
        assert roi_from_dict.display_name == roi.display_name
        assert roi_from_dict.unique_name != roi.unique_name

    def test_copy(self, roi_with_physical_coords):
        # When
        roi = roi_with_physical_coords
        # Then
        roi_copy = roi.__copy__()
        # Expect
        assert roi_copy is not roi  # Ensure it's a different instance
        assert roi_copy.x_pixel_start == roi.x_pixel_start
        assert roi_copy.x_pixel_end == roi.x_pixel_end
        assert roi_copy.y_pixel_start == roi.y_pixel_start
        assert roi_copy.y_pixel_end == roi.y_pixel_end
        assert roi_copy._has_physical_coords == roi._has_physical_coords
        if roi._has_physical_coords:
            assert sc.identical(roi_copy.x_start, roi.x_start)
            assert sc.identical(roi_copy.x_end, roi.x_end)
            assert sc.identical(roi_copy.y_start, roi.y_start)
            assert sc.identical(roi_copy.y_end, roi.y_end)
        assert roi_copy.display_name == roi.display_name
        assert roi_copy.unique_name != roi.unique_name  # unique_name should be different in the copy
