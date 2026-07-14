# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import plopp as pp
import scipp as sc
from easyscience.base_classes import EasyList
from easyscience.base_classes import ModelBase
from easyscience.base_classes import NewBase
from scipp import DimensionError
from scipp import UnitError


class SampleModel(ModelBase):
    """
    Some text
    """


def __init__(self, lattice, more):
    self._lattice = lattice
