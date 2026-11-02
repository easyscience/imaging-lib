# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .scene import build_scene

if TYPE_CHECKING:
    from ..lattice import Lattice

# The web page that draws the lattice with three.js. Its placeholders THREE_JS_URL, ORBIT_CONTROLS_URL,
# CSS2D_RENDERER_URL and LATTICE_SCENE_JSON are replaced by `plot_lattice`.
TEMPLATE = Path(__file__).parent / 'template.html'

# three.js, its mouse controls and its text label renderer, embedded in the page so the view also works without
# internet access
THREE_JS = Path(__file__).parent / 'vendor' / 'three.module.min.js'
ORBIT_CONTROLS_JS = Path(__file__).parent / 'vendor' / 'OrbitControls.js'
CSS2D_RENDERER_JS = Path(__file__).parent / 'vendor' / 'CSS2DRenderer.js'


class LatticeView:
    """An interactive 3D view of a lattice, shown by Jupyter when it is the last expression of a cell.

    Parameters
    ----------
    page : str
        The complete, standalone HTML page that draws the lattice.
    width : int
        The width of the view in pixels.
    height : int
        The height of the view in pixels.
    """

    def __init__(self, page: str, width: int, height: int):
        self.page = page
        self.width = width
        self.height = height

    # This is an IPython specific dunder method which overwrites the __repr__ method.
    def _repr_html_(self) -> str:
        # Show the page in an iframe. An iframe is a web page within a web page.
        return (
            f'<iframe srcdoc="{html.escape(self.page)}" width="{self.width}" height="{self.height}" '
            'style="border: none"></iframe>'
        )


def plot_lattice(lattice: Lattice, width: int = 600, height: int = 400) -> LatticeView:
    """Show an interactive 3D view of a lattice in a Jupyter notebook.

    Drag to rotate, scroll to zoom and right-drag to pan. The buttons change the view and show or hide the atoms,
    their labels, the bonds, the unit cell and the lattice vectors.

    Parameters
    ----------
    lattice : Lattice
        The lattice to show.
    width : int
        The width of the view in pixels.
    height : int
        The height of the view in pixels.

    Returns
    -------
    LatticeView
        The view, which Jupyter shows when it is the last expression of a cell.
        Its ``page`` attribute holds the complete HTML page, which can also be saved and opened in a browser.
    """
    page = TEMPLATE.read_text(encoding='utf-8')
    page = page.replace('THREE_JS_URL', _javascript_data_url(THREE_JS))
    page = page.replace('ORBIT_CONTROLS_URL', _javascript_data_url(ORBIT_CONTROLS_JS))
    page = page.replace('CSS2D_RENDERER_URL', _javascript_data_url(CSS2D_RENDERER_JS))
    page = page.replace('LATTICE_SCENE_JSON', json.dumps(build_scene(lattice)))
    return LatticeView(page, width, height)


def _javascript_data_url(path: Path) -> str:
    """Return the content of a JavaScript file as a base64-encoded ``data:`` URL, so the browser can load it
    without internet access."""
    encoded_content = str(base64.b64encode(path.read_bytes()), encoding='ascii')
    return f'data:text/javascript;base64,{encoded_content}'
