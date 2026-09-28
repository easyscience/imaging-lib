---
icon: material/code-braces-box
---

# :material-code-braces-box: API Reference

This section contains the reference detailing the functions and modules
available in EasyImaging.

- [measurement](measurement/measurement.md) - Loads and inspects
  measurement data.
- [regions_of_interest](regions_of_interest/index.md) - The module
  containing all types of Regions Of Interests (ROIs) available:
    - [RectangleROI](regions_of_interest/rectangle_roi.md) - A rectangular
    ROI. The most basic and common type of ROI
- [sample_model](sample_model/index.md) - The module containing the classes
  and functions related to creating a model of the sample:
    - [Lattice](sample_model/lattice.md) - The crystallographic unit cell of the sample model.
    - [AtomSite](sample_model/atom_site.md) - The atomic sites of a crystallographic lattice.
    - [Atoms](sample_model/atoms.md) - The atomic species occupying an atomic site.
    - [crystals](sample_model/crystals) - A module containing convenience functions to easily
      create and occupy common crystallographic lattices.
