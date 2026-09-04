# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from easyscience import global_object
from gemmi import cif


def find_value_in_cif_block(cif_block: cif.Block, key: str) -> float | None:
    search_result = cif_block.find_values(key)
    if not search_result:
        search_result = cif_block.find_values(key.replace('.', '_'))
    if not search_result:
        return None
    if len(search_result) > 1:
        global_object.logger.warning(f'Multiple values found for key "{key}" in CIF block. Using the first one.')
    string_value = search_result[0].split('(')[0].strip()  # Remove any uncertainty notation
    try:
        return float(string_value)
    except ValueError:
        raise ValueError(f'Value for key "{key}" in CIF block is not a valid float: {string_value}')

def find_loop_in_cif_block(
        cif_block: cif.Block,
        keys: list[str],
        prefix: str | None = None
        ) -> list[dict[str, float | str | None]]:
    if prefix:
        table = cif_block.find(prefix, keys)
        if not table:
            table = cif_block.find(prefix.replace('.', '_'), keys)
        if not table:
            return []
    else:
        table = cif_block.find(keys)
        if not table:
            table = cif_block.find([key.replace('.', '_') for key in keys])
        if not table:
            return []
    result = []
    for row in table:
        row_dict = {}
        for key, value in zip(keys, row):
            try:
                row_dict[key] = float(value)
            except ValueError:
                row_dict[key] = value if value else None
        result.append(row_dict)
    return result

def cif_loader(file_path: str | Path, block: int = 0) -> dict:
    file_path = str(file_path)
    try:
        cif_file = cif.read_file(file_path)
    except Exception as e:
        raise ValueError(f'Failed to read CIF file at {file_path}. Error: {e}')
    if abs(block) >= len(cif_file):
        raise ValueError(f'Block index {block} is out of range for CIF file with {len(cif_file)} blocks.')
    cif_block = cif_file[block]



    return {
        'block_name': cif_block.name,
        'length_a': find_value_in_cif_block(cif_block, '_cell.length_a'),
        'length_b': find_value_in_cif_block(cif_block, '_cell.length_b'),
        'length_c': find_value_in_cif_block(cif_block, '_cell.length_c'),
        'alpha': find_value_in_cif_block(cif_block, '_cell.angle_alpha'),
        'beta': find_value_in_cif_block(cif_block, '_cell.angle_beta'),
        'gamma': find_value_in_cif_block(cif_block, '_cell.angle_gamma'),
    }
