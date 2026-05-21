# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


import pooch

BRAIN = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache('easyimaging'),
    # The remote data is on Github
    base_url='https://raw.githubusercontent.com/easyscience/imaging/refs/heads/master/data/',
    version=None,
    # If this is a development version, get the data from the "main" branch
    version_dev='master',
    registry={
        'iron_alpha_scitiff/iron_alpha_v1.tiff': 'sha256:efb923947e5f5c0f46788c0f18836a3fce916745a05ad5fb8483e5ff17e08a1a',
        'iron_alpha_tiff/iron_alpha_v1.tiff': 'sha256:a1e4a14bfff3d7c0e92cc42f65f7931536d01f763ba7df2a0e4c0aac851f3594',
    },
)


def iron_alpha_scitiff():
    """
    Load the Iron Alpha sample data as a scitiff.
    """
    fname = BRAIN.fetch('iron_alpha_scitiff/iron_alpha_v1.tiff')
    return fname


def iron_alpha_tiff():
    """
    Load the Iron Alpha sample data as a tiff.
    """
    fname = BRAIN.fetch('iron_alpha_tiff/iron_alpha_v1.tiff')
    return fname
