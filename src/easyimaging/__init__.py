# SPDX-FileCopyrightText: 2024 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

"""EasyImaging library."""

from importlib.metadata import version

from .measurement import Measurement
from .sample_model.sample_model import SampleModel

__version__ = version(__package__)
