# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import scipp as sc


def _to_edges(centers: sc.Variable) -> sc.Variable:
    """Convenience method to convert center coordinates to edge coordinates."""
    interior_edges = sc.midpoints(centers)
    return sc.concat(
        [
            2 * centers[0] - interior_edges[0],
            interior_edges,
            2 * centers[-1] - interior_edges[-1],
        ],
        dim=centers.dim,
    )
