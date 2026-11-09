# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive 3D view of a [`Lattice`][...lattice.Lattice] in Jupyter notebooks, drawn with three.js like the structure
view in EasyDiffraction. three.js is embedded in the output, so the view also works without internet access."""

from .viewer import LatticeView
from .viewer import plot_lattice
