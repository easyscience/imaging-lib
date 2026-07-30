# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import math
from typing import TYPE_CHECKING
from typing import Sequence

from easyscience import global_object
from easyscience.base_classes import ModelBase

from easyimaging import Parameter

from ..utils import generate_unique_name_no_zero
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
            force_normalization: bool = False,
            unique_name: str | None = None,
            display_name: str | None = None,
    ):
        """
        Initialize a Mixture instance.

        Parameters
        ----------
        components : Sequence[Sequence[Lattice, int | float]] | None
            A sequence of 2-element sequences containing [`Lattice`][..lattice] objects and their respective mixing fractions.
        force_normalization : bool
            Whether to force normalization of the component fractions.
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

        if force_normalization:
            self.force_normalization = force_normalization  # Use the setter logic
        else:
            self._force_normalization = force_normalization

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
    def force_normalization(self) -> bool:
        """
        Whether to automatically enforce that the Mixtures fractions sum up to 1 always.

        Returns
        -------
        bool
            True if the Mixture is set to force normalization of component fractions, False otherwise.
        """
        return self._force_normalization

    @force_normalization.setter
    def force_normalization(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f'force_normalization must be a boolean, got {type(value)}')
        if value == self._force_normalization:
            return  # No change, do nothing
        if value:
            free_fractions = [fraction for fraction in self._fractions if not fraction.fixed and fraction.independent]
            if len(free_fractions) == 0:
                global_object.log.warning('force_normalization is set to True, but there are no free components. '
                'Normalization cannot be enforced.')
            else:
                self.normalize_fractions()
            for fraction in self._fractions:
                fraction._attach_observer(self)
                # If the Parameter was not a dependent parameter, delete the created serializer ID
                # to avoid messing up the serialization/deserialization.
                if len(fraction._observers) == 1:
                    del fraction.__serializer_id
        else:
            for fraction in self._fractions:
                if len(fraction._observers) == 1 and fraction._observers[0] is self:
                    fraction.__serializer_id = 'temp'  # this is needed to avoid an error since _detach_observer deletes it
                fraction._detach_observer(self)
        self._force_normalization = value

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
        global_object.log.warning('Normalization did not succeed, possibly due to Dependent Parameters.'
                                      f'Total fraction after normalization: {total_fraction_after}.'
                                      ' Trying again updating only the unobserved fractions, if any.')
        unobserved_fractions = [fraction for fraction in independent_fractions if len(fraction._observers) == 0]
        if len(unobserved_fractions) == 0:
            global_object.log.warning('All independent fractions are being observed.'
            'Normalization could not be completed.')
            return
        remainder = 1.0 - sum(fraction.value for fraction in self._fractions)
        self._distribute_remainder(remainder, unobserved_fractions)
        total_fraction_after = sum(fraction.value for fraction in self._fractions)
        if not math.isclose(total_fraction_after, 1.0, rel_tol=1e-3):
            global_object.log.warning('Normalization still did not succeed.'
                                      f'Total fraction after second normalization: {total_fraction_after}.')

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
            variables.extend(lattice.get_all_variables())
            variables.append(self._fractions[index])
        return variables

    def _update(self):
        """
        This method uses the observer pattern of the dependent Parameter implementation to enforce the normalization of
        mixture fractions when force_normalization is True.
        """
        total_fraction = sum(fraction.value for fraction in self._fractions)
        if total_fraction == 1.0:
            return
        fractions = self.fractions
        # Get the object which called this method
        caller_frame = inspect.stack()[1].frame
        local_variables = caller_frame.f_locals
        caller = local_variables['self']
        # Only re-normalize the other Parameters
        fractions.remove(caller)
        for fraction in fractions:
            fraction._scalar.value = fraction.value + (total_fraction - 1) / len(fractions)
        # Update values first, then notify potential observers to avoid infinite loop
        for fraction in fractions:
            fraction._notify_observers()

    def _create_fraction_parameter(self, fraction: Numeric, lattice: Lattice) -> Parameter:
        unique_name = generate_unique_name_no_zero(f'{self.unique_name}_{lattice.unique_name}_fraction')
        parameter = Parameter(value=fraction, min=0.0, max=1.0, fixed=True, unique_name=unique_name)
        parameter._default_unique_name = True  # This gets set to False by the super init
        return parameter
