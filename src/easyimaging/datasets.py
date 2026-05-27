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
        'iron_alpha_scitiff/iron_alpha_v1.tiff': 'sha256:ada3d50a5324362add0b8787e11b87b5f33e1dd7e3862f92face03671a20a282',
        'iron_alpha_tiff/iron_alpha_v1.tiff': 'sha256:a1e4a14bfff3d7c0e92cc42f65f7931536d01f763ba7df2a0e4c0aac851f3594',
        'small_test_scitiff/small_test_scitiff_v1.tiff': (
            'sha256:2ce4088fad10437a1180855fba4ebfefad0c98b252be36d6fc046a1aa6ca922d'
        ),
        'small_test_tiff/small_test_tiff_v1.tiff': 'sha256:99e5558db28d0d5d5ca46d159791b3649d5a61b6cb4ff67bedaf5485b7d6ea35',
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


def small_test_scitiff():
    """
    Load a small test scitiff file.
    """
    fname = BRAIN.fetch('small_test_scitiff/small_test_scitiff_v1.tiff')
    return fname


def small_test_tiff():
    """
    Load a small test tiff file.
    """
    fname = BRAIN.fetch('small_test_tiff/small_test_tiff_v1.tiff')
    return fname
