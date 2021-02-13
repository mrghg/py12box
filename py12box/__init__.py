from pathlib import Path

_ROOT = Path(__file__).parent

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
    
    data_path = _ROOT / "data" / sub_path

    return data_path
