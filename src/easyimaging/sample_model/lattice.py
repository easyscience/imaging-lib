# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object
from easyscience.base_classes import ModelBase
from easyscience.variable import Parameter

Numeric = int | float


class Lattice(ModelBase):
    """
    A Lattice represents the periodic arrangement of atoms in a crystal structure, defined by its lattice parameters.
    """

    def __init__(
            self,
            length_a: Numeric,
            length_b: Numeric,
            length_c: Numeric,
            alpha: Numeric,
            beta: Numeric,
            gamma: Numeric,
            unique_name: str | None = None,
            display_name: str | None = None
            ):
        """
        Initialize a Lattice instance.

        Parameters
        ----------
        length_a : float | int
            The length of the lattice vector along the x-axis.
        length_b : float | int
            The length of the lattice vector along the y-axis.
        length_c : float | int
            The length of the lattice vector along the z-axis.
        alpha : float | int
            The angle between the b and c lattice vectors (in degrees).
        beta : float | int
            The angle between the a and c lattice vectors (in degrees).
        gamma : float | int
            The angle between the a and b lattice vectors (in degrees).
        unique_name : str | None
            A unique identifier for the [`Lattice`][..]. Defaults to a generated name if not provided.
        display_name : str | None
            A prettily formatted name for the [`Lattice`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
        """

