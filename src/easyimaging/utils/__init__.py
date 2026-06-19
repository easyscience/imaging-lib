# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from .jupyter_notebook_utilities import _is_notebook
from .scipp_utilities import _to_edges

__all__ = [_to_edges, _is_notebook]
