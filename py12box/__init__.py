from pathlib import Path as _pth

__version__ = "0.1.1"

_ROOT = _pth(__file__).parent

def get_data(sub_path):
    """Get path to data files

    Parameters
    ----------
    sub_path : str
        path to data files, relative to py12box/data directory

    Returns
    -------
    pathlib.Path
        pathlib Path to data folder/file
    
    """
    
    if sub_path[0] == "/":
        raise Exception("sub-path can't begin with '/'")

    data_path = _ROOT / "data" / sub_path

    return data_path

from . import model
from . import core
from . import startup
from . import util