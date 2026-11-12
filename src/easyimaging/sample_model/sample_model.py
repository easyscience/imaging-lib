# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numbers
from typing import Sequence

import numpy as np
import scipp as sc
from easyscience.base_classes import ModelBase
from scipp import UnitError

from easyimaging.utils.jupyter_notebook_utilities import _is_notebook

from .calculators import NCrystalCalculator
from .lattice import Lattice
from .sample_properties import SampleProperties


class SampleModel(ModelBase):
    """The sample model for representing a sample in neutron imaging experiments."""

    def __init__(
        self,
        lattice: Lattice,
        sample_properties: SampleProperties,
        unique_name: str | None = None,
        display_name: str | None = None,
    ):
        """Initialize the sample model with a lattice and sample properties."""
        if not isinstance(lattice, Lattice):
            raise TypeError('lattice must be an instance of Lattice')
        if not isinstance(sample_properties, SampleProperties):
            raise TypeError('sample_properties must be an instance of SampleProperties')
        super().__init__(unique_name=unique_name, display_name=display_name)
        self._lattice = lattice
        self._sample_properties = sample_properties

    @property
    def lattice(self) -> Lattice:
        """Get the lattice of the sample."""
        return self._lattice

    @property
    def sample_properties(self) -> SampleProperties:
        """Get the sample properties of the sample."""
        return self._sample_properties

    @lattice.setter
    def lattice(self, lattice: Lattice):
        """Set the lattice of the sample."""
        if not isinstance(lattice, Lattice):
            raise TypeError('lattice must be an instance of Lattice')
        self._lattice = lattice

    @sample_properties.setter
    def sample_properties(self, sample_properties: SampleProperties):
        """Set the sample properties of the sample."""
        if not isinstance(sample_properties, SampleProperties):
            raise TypeError('sample_properties must be an instance of SampleProperties')
        self._sample_properties = sample_properties

    def display(self):
        """
        Display a 3D visualization of the `SampleModel`s lattice.

        Atoms are represented as spheres, with their size and color corresponding to the element type.<br>
        Atoms on the unit cell boundaries are mirrored on the opposite sides of the unit cell.

        Bonds between atoms are shown as cylinders and are rendered only if the distance between them is within the atoms
        covalent radii sum plus a small 0.25 Å threshold **and** lie within the first coordination shell of both atoms.

        ???+ info "Controls"
            **Camera controls**

            - Left-click and drag to rotate the camera
            - Wheel-scroll to zoom in and out
            - Right-click and drag to pan the camera

            **Buttons**

            &ensp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="4.5"
              width="17" height="15" rx="1.5"/><path d="M9.5 4.5V19.5M14.5 4.5V19.5"/></svg> /
            <svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3.5" y="4.5"
              width="17" height="15" rx="1.5"/><path d="M9 4.5 11 19.5M15 4.5 13 19.5"/></svg>
            &ensp;&thinsp; Change camera perspective between parallel and orthographic view <br>
            &ensp;<b style="color: #DC2828">a</b> / <b style="color: #28B428">b</b> / <b style="color: #2850DC">c</b>
            &nbsp;&thinsp; Change camera view to view along the corresponding axis <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 11 12 3l9 8"/><path d="M5 9.5V20h5v-6h4v6h5V9.5"/></svg>
            &ensp;&emsp; Reset camera view to the default orientation <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><path d="M11.8 12.2 14 10M12.9 16.1l1.8.4"/>
              <circle cx="9" cy="15" r="4.2" fill="currentColor" stroke="none"/><circle cx="16" cy="8" r="3"
              fill="currentColor" stroke="none"/><circle cx="17" cy="17" r="2.6" fill="currentColor" stroke="none"/></svg>
            &ensp;&emsp; Toggle visibility of atoms <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"><rect x="3.5" y="5"
              width="17" height="14" rx="2"/><path d="M9 15.5 12 8.5l3 7M10.1 13.2h3.8"/></svg>
            &ensp;&emsp; Toggle visibility of atomic labels <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24"><line x1="8"
              y1="16" x2="16" y2="8" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><circle cx="6.7"
              cy="17.3" r="3.1" fill="currentColor"/><circle cx="17.3" cy="6.7" r="3.1" fill="currentColor"/></svg>
            &ensp;&emsp; Toggle visibility of chemical bonds <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round">
              <path d="M12 2.5 20 7v9l-8 4.5L4 16V7Z"/><path d="M4 7l8 4.5L20 7M12 11.5v9"/></svg>
            &ensp;&emsp; Toggle visibility of the unit cell boundaries <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
              <path d="M5 19V6M5 19h13M5 19l7-7"/><path d="M5 6 3.3 8.2M5 6l1.7 2.2M18 19l-2.2-1.4M18 19l-2.2 1.4"/></svg>
            &ensp;&emsp; Toggle visibility of the lattice vectors <br>
            &ensp;&emsp;<svg width='1.2em' height='1.2em' style='vertical-align: -0.25em' viewBox="0 0 24 24" fill="none"
              stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z"/><circle cx="12"
              cy="13" r="3.4"/></svg>
            &ensp;&emsp; Save the current view of the lattice as a PNG


        ```python exec="true"
        from easyimaging.sample_model import Atoms
        from easyimaging.sample_model.crystals import body_centered_cubic

        lattice = body_centered_cubic(
            length_a=2.87,
            atom=Atoms.Fe,
            debye_temperature=400.0,
        )
        view = lattice.display()._repr_html_()
        print(
            '<div style="display: flex; width: fit-content; margin:0 auto; '
            f'border: .05rem solid var(--md-typeset-table-color); border-radius: .1rem">{view}</div>'
        )
        ```

        Raises
        ------
        LatticeError
            If the lattice structure is invalid and therefore cannot be displayed.
        """
        return self.lattice.display()

    def spectrum(self, wavelengths: np.ndarray | Sequence[numbers.Real] | sc.Variable) -> sc.DataArray:
        """
        Calculate the spectrum of the sample for the given wavelengths.

        If the wavelengths are provided as a sc.Variable array, it must have units of length.<br>
        If a np.ndarray or list/tuple of values is provided, the wavelengths will be assumed to be in units of Å.

        Parameters
        ----------
        wavelengths : np.ndarray | Sequence[float | int] | sc.Variable
            The wavelengths at which to calculate the scattering intensity.

        Returns
        -------
        sc.DataArray
            The calculated scattering intensity for each wavelength.
        """
        if isinstance(wavelengths, np.ndarray):
            internal_wavelengths = wavelengths
        elif isinstance(wavelengths, sc.Variable):
            try:
                internal_wavelengths = wavelengths.to(unit='angstrom').values
            except UnitError as e:
                raise UnitError('"wavelengths" must have units of length') from e
        elif isinstance(wavelengths, Sequence):
            internal_wavelengths = np.array(wavelengths)
        else:
            raise TypeError('wavelengths must be a np.ndarray, a sc.Variable array, or a Sequence of values')
        if internal_wavelengths.ndim != 1:
            raise ValueError('wavelengths must be a one-dimensional array')
        if not np.isdtype(internal_wavelengths.dtype, ('real floating', 'integral')):
            raise TypeError('wavelengths must be of a numeric type')
        if np.any(internal_wavelengths <= 0):
            raise ValueError('wavelengths must be positive')
        self.calculator = NCrystalCalculator(lattice=self.lattice, sample_properties=self.sample_properties)
        calculated_spectrum = sc.array(dims=['t'], values=self.calculator.calculate(internal_wavelengths))
        wavelengths_coord = sc.array(dims=['t'], values=internal_wavelengths, unit='angstrom')
        return sc.DataArray(data=calculated_spectrum, coords={'wavelength': wavelengths_coord})

    def plot(self, wavelengths: np.ndarray | Sequence[numbers.Real] | sc.Variable | None = None, **kwargs) -> None:
        """
        Plot the spectrum of the sample for the given wavelengths.

        If the wavelengths are provided as a sc.Variable array, it must have units of length.<br>
        If a np.ndarray or list/tuple of values is provided, the wavelengths will be assumed to be in units of Å.
        If no wavelengths are provided, a default range from 0.5 Å to 8 Å will be used.

        This method uses the [plopp](https://scipp.github.io/plopp/plotting/line-plot.html) library for plotting.

        To customize the plot appearance, additional keyword arguments can be passed to the underlying plopp plotting function.
         See [plopp.plot][] for which keyword arguments are available and how to use them.

        Parameters
        ----------
        wavelengths : np.ndarray | Sequence[float | int] | sc.Variable | None
            The wavelengths at which to calculate the scattering intensity.
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function.<br>
            See [plopp.plot][] for options.
        """
        if wavelengths is None:
            wavelengths = np.linspace(0.5, 8, 401)
        spectrum_data = self.spectrum(wavelengths)

        plot_kwargs_defaults = {
            'title': self.unique_name + ' - Transmission spectrum',
            'xlabel': f'Wavelength [{spectrum_data.coords["wavelength"].unit}]',
            'ylabel': 'Transmission',
            'ymin': 0.0,
            'ymax': min(1.1, float(spectrum_data.max().value * 1.1)),
            'coords': ['wavelength'],
            'linestyle': 'solid',
            'marker': None,
        }
        # Overwrite defaults with any user-provided kwargs
        plot_kwargs_defaults.update(kwargs)

        if _is_notebook():
            return spectrum_data.plot(**plot_kwargs_defaults)
        else:
            plot = spectrum_data.plot(**plot_kwargs_defaults)
            plot.show()
