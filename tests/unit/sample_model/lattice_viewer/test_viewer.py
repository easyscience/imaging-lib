# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import base64

import pytest

from easyimaging.sample_model import crystals
from easyimaging.sample_model._lattice_viewer import viewer


@pytest.fixture
def lattice():
    return crystals.body_centered_cubic(length_a=2.0, atom='Fe', debye_temperature=300)


def test_javascript_data_url(tmp_path):
    # When
    path = tmp_path / 'library.js'
    path.write_text('const answer = 42')
    # Then
    url = viewer._javascript_data_url(path)
    # Expect
    prefix = 'data:text/javascript;base64,'
    assert url.startswith(prefix)
    assert base64.b64decode(url[len(prefix) :]) == b'const answer = 42'


def test_plot_lattice(lattice):
    # When Then
    view = viewer.plot_lattice(lattice, width=500, height=300)
    # Expect
    for javascript_file in (viewer.THREE_JS, viewer.ORBIT_CONTROLS_JS, viewer.CSS2D_RENDERER_JS):
        assert viewer._javascript_data_url(javascript_file) in view.page  # Works without internet access
    assert view.page.count('"element": "Fe"') == 9  # 8 corner atoms and 1 body-centre atom
    for label in ('a', 'b', 'c'):
        assert f'"label": "{label}"' in view.page
    for placeholder in ('THREE_JS_URL', 'ORBIT_CONTROLS_URL', 'CSS2D_RENDERER_URL', 'LATTICE_SCENE_JSON'):
        assert placeholder not in view.page
