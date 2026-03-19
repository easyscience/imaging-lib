
def _is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook environment.
    """
    try:
        shell = get_ipython().__class__.__name__  # pyright: ignore[reportUndefinedVariable]
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (possibly other IDE)
    except NameError:
        return False  # Probably standard Python interpreter
    