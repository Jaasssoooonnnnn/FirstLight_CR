import pytest


@pytest.fixture(scope="session")
def desktop_root():
    """One Tcl interpreter for widget tests, never a game or native runtime."""
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as error:
        if "display" in str(error).lower():
            pytest.skip("A desktop display is required for Tk widget tests")
        raise
    root.withdraw()
    yield root
    root.destroy()
