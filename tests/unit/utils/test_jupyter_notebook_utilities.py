# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import builtins

import pytest

from easyimaging.utils import _is_notebook


class FakeShell:
    """Stand-in for an IPython shell, identified by its class name."""


class ZMQInteractiveShell(FakeShell):
    pass


class TerminalInteractiveShell(FakeShell):
    pass


class TestIsNotebook:
    @pytest.fixture(autouse=True)
    def _no_injected_get_ipython(self, monkeypatch):
        """
        IPython only injects `get_ipython` into the builtins while a cell is executing, so it is absent whenever
        the library is called from a widget callback. Remove it to test under those conditions.
        """
        monkeypatch.delattr(builtins, 'get_ipython', raising=False)

    @pytest.mark.parametrize(
        ('shell', 'expected'),
        [
            (ZMQInteractiveShell(), True),  # Jupyter notebook, Voila or qtconsole
            (TerminalInteractiveShell(), False),  # Terminal running IPython
            (None, False),  # Standard Python interpreter
        ],
    )
    def test_detects_shell(self, monkeypatch, shell, expected):
        monkeypatch.setattr('IPython.core.getipython.get_ipython', lambda: shell)
        assert _is_notebook() is expected
