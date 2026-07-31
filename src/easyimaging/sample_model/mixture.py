# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import TYPE_CHECKING
from typing import Sequence

from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import ModelBase

from .lattice import Lattice

if TYPE_CHECKING:
    from easyscience.variable import DescriptorNumber

Numeric = int | float


class Mixture(ModelBase):
    """
    A Mixture is a combination of two or more substances, each represented by a [`Lattice`][..lattice] object.<br>
    The Mixture class allows for the representation of complex materials with multiple phases or components.
    """

    def __init__(
        self,
        components: Sequence[Sequence[Lattice, Numeric]] = None,
        auto_normalize: bool = False,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """
        Initialize a Mixture instance.

        Parameters
        ----------
        components : Sequence[Sequence[Lattice, int | float]] | None
            A sequence of 2-element sequences containing [`Lattice`][..lattice] objects and their respective mixing fractions.
        auto_normalize : bool
            Whether to use automatically normalized component fractions for calculations.
        unique_name : str | None
            A unique identifier for the [`Mixture`][..]. Defaults to ``'Mixture'`` appended by a unique integer.
        display_name : str | None
            A prettily formatted name for the [`Mixture`][..]. Defaults to [`unique_name`][..unique_name] if not provided.
        """
        super().__init__(unique_name=unique_name, display_name=display_name)

        self._components = []
        self._fractions = []

        if not isinstance(components, (Sequence, None)):
            raise TypeError(f'components must be a sequence of tuples, got {type(components)}')
        for component in components:
            if not isinstance(component, Sequence) or len(component) != 2:
                raise TypeError(f'Each component must be a sequence of length 2, got {component}')
            lattice, fraction = component
            if not isinstance(lattice, Lattice):
                raise TypeError(f'First element of each component must be a Lattice object, got {type(lattice)}')
            if not isinstance(fraction, Numeric):
                raise TypeError(f'Second element of each component must be a numeric value, got {type(fraction)}')
            self._components.append(lattice)
            self._fractions.append(self._create_fraction_parameter(fraction, lattice))

        self._auto_normalize = auto_normalize
        self._normalized_fractions = []

    def _create_fraction_parameter(self, fraction: Numeric, lattice: Lattice) -> Parameter:
        unique_name = global_object.generate_unique_name(f'{self.unique_name}_{lattice.unique_name}_fraction')
        parameter = Parameter(value=fraction, min=0.0, max=1.0, fixed=True, unique_name=unique_name)
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter

    @property
    def components(self) -> list[Lattice]:
        """
        Get the list of [`Lattice`][..lattice] components in the Mixture.

        Note
        ----
        The returned list is a copy of the internal list of components. Modifying this list will not affect the Mixture's
        components.

        Returns
        -------
        list[Lattice]
            A list of [`Lattice`][..lattice] objects representing the components of the Mixture.
        """
        return list(self._components)

    @property
    def fractions(self) -> list[Parameter]:
        """
        Get the list of mixing fractions for each component in the Mixture.

        Note
        ----
        The returned list is a copy of the internal list of fractions. Modifying this list will not affect the Mixture's
        fractions.

        Returns
        -------
        list[Parameter]
            A list of [`Parameter`][..] objects representing the mixing fractions of each component in the Mixture.
        """
        return list(self._fractions)

    @property
    def auto_normalize(self) -> bool:
        """
        Whether to use automatically normalized fractions for the components in the Mixture.
        If True, the defined components given by the fractions property act as scaling factors for the normalized fraction.

        Returns
        -------
        bool
            Whether to use automatically normalized fractions for the components in the Mixture for calculations.
        """
        return self._auto_normalize

    @auto_normalize.setter
    def auto_normalize(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f'auto_normalize must be a boolean, got {type(value)}')
        self._auto_normalize = value

    def display_auto_normalized_fractions(self) -> list[float]:
        """
        Get a list of the current values of the auto-normalized mixing fractions for each component in the Mixture.

        Returns
        -------
        list[float]
            A list of auto-normalized mixing fractions for each component in the Mixture.
        """
        self._create_normalized_fractions()
        return [fraction.value for fraction in self._normalized_fractions]

    @property
    def _fractions(self) -> list[Parameter]:
        """
        This property is used by the calculator to get the correct fractions to use for calculations.
        """
        if self._auto_normalize:
            # Always create new to ensure correctness
            self._create_normalized_fractions()
            return self._normalized_fractions
        return self._fractions

    def _create_normalized_fractions(self):
        for parameter in self._normalized_fractions:
            parameter.make_independent()
        self._normalized_fractions.clear()
        for fraction in self._fractions:
            if not fraction.fixed:
                unique_name = global_object.generate_unique_name(f'Normalized {fraction.unique_name}')
                scaling = ' - '.join(f.unique_name for f in self._fractions if f.fixed)
                denominator = '(' + ' + '.join(f.unique_name for f in self._fractions if not f.fixed) + ')'
                dependency_expression = f'{fraction.unique_name} / {denominator}'
                if scaling:
                    dependency_expression = f'(1.0 - {scaling}) * ({dependency_expression})'
                dependency_map = {f.unique_name: f for f in self._fractions}
                parameter = Parameter.from_dependency(
                    name=unique_name,
                    dependency_expression=dependency_expression,
                    dependency_map=dependency_map,
                    unique_name=unique_name,
                )
            else:
                unique_name = global_object.generate_unique_name(f'{fraction.unique_name} pass through')
                parameter = Parameter.from_dependency(
                    name=unique_name,
                    dependency_expression=f'{fraction.unique_name}',
                    dependency_map={fraction.unique_name: fraction},
                    unique_name=unique_name,
                )
            self._normalized_fractions.append(parameter)

    def add_component(self, lattice: Lattice, fraction: Numeric):
        """
        Add a new component to the Mixture.

        Parameters
        ----------
        lattice : Lattice
            The [`Lattice`][..lattice] object representing the new component.
        fraction : int | float
            The mixing fraction for the new component. Must be between 0.0 and 1.0.

        Raises
        ------
        TypeError
            If `lattice` is not a [`Lattice`][..lattice] object or if `fraction` is not a numeric value.
        ValueError
            If `fraction` is not between 0.0 and 1.0.
        """
        if not isinstance(lattice, Lattice):
            raise TypeError(f'lattice must be a Lattice object, got {type(lattice)}')
        if not isinstance(fraction, Numeric):
            raise TypeError(f'fraction must be a numeric value, got {type(fraction)}')
        if not (0.0 <= fraction <= 1.0):
            raise ValueError('fraction must be between 0.0 and 1.0')
        self._fractions.append(self._create_fraction_parameter(fraction, lattice))
        self._components.append(lattice)

    def remove_component(self, lattice: Lattice | int):
        """
        Remove a component from the Mixture.

        Parameters
        ----------
        lattice : Lattice | int
            The [`Lattice`][..lattice] object or its index in the components list to be removed.

        Raises
        ------
        ValueError
            If `lattice` is not found in the components list.
        IndexError
            If `lattice` is an index that is out of range for the components list.
        TypeError
            If `lattice` is neither a [`Lattice`][..lattice] object nor an integer index.
        """
        if isinstance(lattice, int):
            if 0 <= lattice < len(self._components):
                self._fractions.pop(lattice)
                self._components.pop(lattice)
            else:
                raise IndexError(f'Index {lattice} is out of range for the components list of length {len(self._components)}.')
        elif isinstance(lattice, Lattice):
            try:
                index = self._components.index(lattice)
                self._fractions.pop(index)
                self._components.pop(index)
            except ValueError:
                raise ValueError('The specified Lattice component is not in the Mixture.')
        else:
            raise TypeError(f'lattice must be a Lattice object or an integer index, got {type(lattice)}')
        self._create_normalized_fractions()  # Recreate normalized fractions after removal to avoid stale observer references

    def normalize_fractions(self):
        """
        Normalize the mixing fractions of the components in the Mixture so that they sum to 1.0.

        This method adjusts the values of the [`Parameter`][..] objects representing the mixing fractions of each component
        in the Mixture, ensuring that their total sum equals 1.0.
        """
        total_fraction = sum(fraction.value for fraction in self._fractions)
        if total_fraction == 1.0:
            return
        independent_fractions = [fraction for fraction in self._fractions if fraction.independent]
        if not independent_fractions:
            raise ValueError('Cannot normalize fractions: no independent fractions available.')
        if any(fraction.value < 0.0 for fraction in self._fractions):
            raise ValueError('Cannot normalize fractions: one or more fractions have negative values.')
        remainder = 1.0 - total_fraction
        self._distribute_remainder(remainder, independent_fractions)
        total_fraction_after = sum(fraction.value for fraction in self._fractions)
        if math.isclose(total_fraction_after, 1.0, rel_tol=1e-3):
            return
        global_object.log.warning(
            'Normalization did not succeed, possibly due to Dependent Parameters.'
            f'Total fraction after normalization: {total_fraction_after}.'
            ' Trying again updating only the unobserved fractions, if any.'
        )
        unobserved_fractions = [fraction for fraction in independent_fractions if len(fraction._observers) == 0]
        if len(unobserved_fractions) == 0:
            global_object.log.warning('All independent fractions are being observed.Normalization could not be completed.')
            return
        remainder = 1.0 - sum(fraction.value for fraction in self._fractions)
        self._distribute_remainder(remainder, unobserved_fractions)
        total_fraction_after = sum(fraction.value for fraction in self._fractions)
        if not math.isclose(total_fraction_after, 1.0, rel_tol=1e-3):
            global_object.log.warning(
                f'Normalization still did not succeed.Total fraction after second normalization: {total_fraction_after}.'
            )

    def _distribute_remainder(self, remainder: float, fractions: list[Parameter]):
        sorted_fractions = sorted(fractions, key=lambda f: f.value)
        for i, fraction in enumerate(sorted_fractions):
            remainder_part = remainder / (len(fractions) - i)
            if remainder_part > 0.0 or fraction.value >= abs(remainder_part):
                fraction.value += remainder_part
                remainder -= remainder_part
            else:
                remainder += fraction.value
                fraction.value = 0.0

    def get_all_variables(self) -> list[DescriptorNumber]:
        """
        Get a list of all [`Parameter`][..] objects in the `Mixture`, including those from its components.

        Returns
        -------
        list[DescriptorNumber]
            A list of all [`DescriptorNumber`][..] or [`Parameter`][..] objects in the Mixture.
        """
        variables = []
        for index, lattice in enumerate(self._components):
            variables.append(self._fractions[index])
            variables.extend(lattice.get_all_variables())
        return variables
