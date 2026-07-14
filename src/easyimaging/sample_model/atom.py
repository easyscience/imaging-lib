# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

Numeric = int | float


class Atom(ModelBase):
    """
    An atom
    """

    def __init__(
            self,
            chemical_symbol: str,
            fract_x: Numeric,
            fract_y: Numeric,
            fract_z: Numeric,
            unique_name: str | None = None,
            display_name: str | None = None,
    ):

        self._chemical_symbol = chemical_symbol
        self._fract_x = Parameter(value=fract_x, min=0.0, max=1.0)
