# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

class Lattice(ModelBase):
    """
    A Lattice represents the periodic arrangement of atoms in a crystal structure, defined by its lattice parameters.
    """

    def __init__(self, a: float, b: float, c: float, alpha: float, beta: float, gamma: float, unique_name=None, display_name=None):
        """
        Initialize a Lattice instance.

        Parameters
        ----------
        a : float
            The length of the lattice vector along the x-axis.
        b : float
            The length of the lattice vector along the y-axis.
        c : float
            The length of the lattice vector along the z-axis.
        alpha : float
            The angle between the b and c lattice vectors (in degrees).
        beta : float
            The angle between the a and c lattice vectors (in degrees).
        gamma : float
            The angle between the a and b lattice vectors (in degrees).
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to a generated name if not provided.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
        """