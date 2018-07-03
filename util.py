# -*- coding: utf-8 -*-
"""
util.py

Common processes.

Initial author: Edward Chung (s1765003@sms.ed.ac.uk)
Version History
1.0 20171026    EC  Initial code.
2.0 20180112    EC  Function updates; docstrings update;
                    and numpy save file support.
"""
import configparser
import numpy as np
import pandas as pd
import scipy


def io_r_csv(fpath, fmt="np"):
    '''Read csv file

    Parameters
    ----------
    fpath : file-like object, string, or pathlib.Path
        Path of the data file.
    fmt : {"np", "pd"}, optional
        Format of the output data.

    Returns
    -------
    f : ndarray or DataFrame
        Output data.

    '''
    f = pd.read_csv(fpath, sep=',', header=0, skipinitialspace=True)
    if fmt == "np":
        f = f.values
        f = np.ascontiguousarray(f)
    return f


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


def io_r_idlsave(fpath):
    '''Read IDL save file

    Parameters
    ----------
    fpath : file-like object, string, or pathlib.Path
        Path of the data file.

    Returns
    -------
    f : dict
        Dictionary object containing ndarrays.

    '''
    f = scipy.io.readsav(fpath, python_dict=True)
    return f


def io_w_csv(fpath, var, output_boxes='all', fmt='%16.8f'):
    """Write ndarray to csv file

    fpath : file-like object, string, or pathlib.Path
        Path of the data file.
    var : ndarray
        Input data for the file.
    output_boxes : {'all', 'surface'} or list
        List of boxes to include in the output.
    fmt : str
        Format of the floats.

    """
    n_box = len(var[0])
    if output_boxes == 'all':
        output_boxes = np.arange(0, n_box)
    elif output_boxes == 'surface':
        output_boxes = np.arange(0, 4)

    var = var[..., output_boxes]

    header_str = []
    for i in output_boxes:
        header_str.append('box_{}'.format(i))
    header_str = ','.join(header_str)

    np.savetxt(fpath, var, fmt=fmt, delimiter=',', newline='\n',
               header=header_str, comments='')


def io_w_npy(fpath, var):
    r"""Write ndarray to numpy save file

    fpath : file-like object, string, or pathlib.Path
        Path of the data file.
    var : ndarray
        Input data for the file.

    """
    np.save(fpath, var, allow_pickle=False)


def io_c_csv2npy(fpath1, fpath2):
    r"""Convert csv file to np save file

    fpath1 : file-like object, string, or pathlib.Path
        Input csv file.
    fpath2 : file-like object, string, or pathlib.Path
        Output npy file.

    """
    io_w_npy(fpath2, io_r_csv(fpath1))


def read_config(config_file, env='default paths'):
    """Read config file

    Parameters
    ----------
    config_file : file-like object, string, or pathlib.Path
        Path of the configuration file.
    env : str
        Configuration environment.

    Returns
    -------
    c : dict-like
        Configured objects.

    """
    c = configparser.ConfigParser()
    c.sections()
    c.read(config_file)
    c = c[env]
    return c
