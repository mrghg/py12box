"""
Copyright 2021 Matt Rigby

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2020 Eunchong Chung

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

Common processes and helper functions
"""

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

