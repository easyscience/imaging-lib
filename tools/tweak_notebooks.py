# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause
"""Add a hidden static-plot setup cell to generated documentation notebooks."""

import argparse
from pathlib import Path

import nbformat

SETUP_TAG = 'docs-static-plots'
SNAPSHOT_SETUP = """\
from io import BytesIO
from IPython.display import Image, display
from plopp.backends.matplotlib.figure import MplBaseFig
from plopp.widgets import Box

def display_snapshot(obj):
    if hasattr(obj, 'fig'):
        buffer = BytesIO()
        obj.fig.savefig(buffer, format='png', bbox_inches='tight')
        display(Image(data=buffer.getvalue()))
    else:
        for child in getattr(obj, 'children', ()):
            display_snapshot(child)

formatter = get_ipython().display_formatter.ipython_display_formatter
formatter.for_type(MplBaseFig, display_snapshot)
formatter.for_type(Box, display_snapshot)
"""


def main():
    """Add the hidden display setup to stripped documentation notebooks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    notebooks = []
    for path in args.paths:
        notebooks.extend([path] if path.is_file() else sorted(path.rglob('*.ipynb')))
    notebooks = [path for path in notebooks if '.ipynb_checkpoints' not in path.parts]
    if not notebooks:
        parser.error('No documentation notebooks found.')

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        notebook.cells = [cell for cell in notebook.cells if SETUP_TAG not in cell.metadata.get('tags', [])]
        notebook.cells.insert(
            0,
            nbformat.v4.new_code_cell(
                SNAPSHOT_SETUP,
                metadata={'tags': ['hide-in-docs', SETUP_TAG]},
                id=SETUP_TAG,
            ),
        )
        nbformat.write(notebook, path)


if __name__ == '__main__':
    main()
