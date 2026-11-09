# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import NCrystal as nc
import numpy as np
import scipp as sc

from ..lattice import Lattice
from ..sample_properties import SampleProperties


class NCrystalCalculator:
    """The NCrystal calculator for handling neutron scattering calculations."""
    def __init__(
        self,
        lattice: Lattice,
        sample_properties: SampleProperties
    ):
        if not isinstance(lattice, Lattice):
            raise TypeError("lattice must be an instance of Lattice")
        if not isinstance(sample_properties, SampleProperties):
            raise TypeError("sample_properties must be an instance of SampleProperties")
        self.lattice = lattice
        self.sample_properties = sample_properties
        self.setup_material(self.lattice, self.sample_properties)

    def setup_material(self, lattice, sample_properties):
        material = nc.NCMATComposer()
        # Set up the NCrystal lattice
        # Ensure all lattice parameters are converted to the correct units for NCrystal
        length_a = lattice.length_a.full_value.to(unit='Å').value
        length_b = lattice.length_b.full_value.to(unit='Å').value
        length_c = lattice.length_c.full_value.to(unit='Å').value
        alpha = lattice.alpha.full_value.to(unit='deg').value
        beta = lattice.beta.full_value.to(unit='deg').value
        gamma = lattice.gamma.full_value.to(unit='deg').value
        _, spacegroup = lattice.spacegroup
        material.set_cellsg(
            a=length_a,
            b=length_b,
            c=length_c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            spacegroup=spacegroup
        )
        # Fill it with the atoms. Ensure that identical atoms have identical labels.
        # atoms = {}
        # for atom_site in lattice.atom_sites:
        #     distinction = (atom_site.atom.name, atom_site.debye_temperature.full_value.to(unit='K').value)
        #     position = atom_site.fract_a.value, atom_site.fract_b.value, atom_site.fract_c.value
        #     if distinction not in atoms:
        #         atoms[distinction] = [position]
        #     else:
        #         atoms[distinction].append(position)
        atoms = [(atom_site.atom.name + "_" + str(atom_site.debye_temperature.full_value.to(unit='K').value),
                  atom_site.fract_a.value,
                  atom_site.fract_b.value,
                  atom_site.fract_c.value)
                 for atom_site in lattice.atom_sites]
        material.set_atompos(atoms)
        # Set the dynamics and temperature
        temperature = sample_properties.temperature.full_value.to(unit='K').value
        material.set_default_temperature(temperature)
        unique_labels = set(atoms[i][0] for i in range(len(atoms)))
        for label in unique_labels:
            atom_name, debye_temp = label.rsplit("_", 1)
            material.set_dyninfo_vdosdebye(
                label=label,
                debye_temp=float(debye_temp)
            )
            material.set_composition(label, atom_name)
        self.material = material
        self.scatter = material.load()

    def calculate(self, wavelengths: np.ndarray):
        number_density = sc.scalar(self.scatter.info.getNumberDensity(), unit='1/angstrom^3')
        cross_sections = sc.array(
            values=self.scatter.scatter.xsect(wl=wavelengths) + self.scatter.scatter.absorption(wl=wavelengths),
            dims=['wavelength'],
            unit='barn',
        )
        sample_thickness = sc.scalar(
            self.sample_properties.sample_thickness.full_value.to(unit='angstrom').value,
            unit='angstrom',
            )
        spectrum = np.exp((-number_density * cross_sections * sample_thickness).to(unit='1').values)
        return spectrum
