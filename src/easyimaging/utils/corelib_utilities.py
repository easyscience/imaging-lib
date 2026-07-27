# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from easyscience import global_object


def generate_unique_name_no_zero(name_prefix: str) -> str:
    """
    Temporary workaround until this method is changed in EasyScience
    """
    names_with_prefix = [name for name in global_object.map.vertices() if name.startswith(name_prefix + '_')]
    if names_with_prefix:
        name_with_prefix_count = [0]
        for name in names_with_prefix:
            # Strip away the prefix and trailing _
            name_without_prefix = name.replace(name_prefix + '_', '')
            if name_without_prefix.isdecimal():
                name_with_prefix_count.append(int(name_without_prefix))
        unique_name = f'{name_prefix}_{max(name_with_prefix_count) + 1}'
    else:
        unique_name = f'{name_prefix}_0'
    return unique_name
