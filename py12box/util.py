import numpy as np
import pandas as pd
from pathlib import Path


def io_r_npy(fpath, mmap_mode='r'):
    '''Read npy file

    Parameters
    ----------
    fpath : file-like object, string, or pathlib.Path
        Path of the data file.

    Returns
    -------
    f : ndarray
        Output data.

    '''
    f = np.load(fpath, mmap_mode=mmap_mode, allow_pickle=False)
    f = np.ascontiguousarray(f)
    return f


def io_r_npz(fpath):
    '''Read npz file

    Parameters
    ----------
    fpath : file-like object, string, or pathlib.Path
        Path of the data file.

    Returns
    -------
    f : ndarray
        Output data.

    '''
    d = np.load(fpath, mmap_mode='r', allow_pickle=False)
    for key in d.keys():
        f = np.ascontiguousarray(d[key])
        yield f

