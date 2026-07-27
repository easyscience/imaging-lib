# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from easyscience.base_classes import ModelBase


class SampleModel(ModelBase):
    """
    Some text
    """


def __init__(self, lattice, more):
    self._lattice = lattice
