# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause


def _is_notebook() -> bool:
    """Check if the code is running in a Jupyter notebook environment."""
    # IPython only injects `get_ipython` into the builtins while a cell is executing, so it must be imported here
    # instead: otherwise this returns False when called from a widget callback, such as a button click in a dashboard.
    try:
        from IPython.core.getipython import get_ipython
    except ImportError:
        return False  # Without IPython it cannot be a notebook

    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        return True  # Jupyter notebook, Voila, qtconsole or similar
    elif shell == 'TerminalInteractiveShell':
        return False  # Terminal running IPython
    else:
        return False  # Standard Python interpreter ('NoneType') or other type (possibly other IDE)
