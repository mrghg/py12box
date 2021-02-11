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

from py12box import startup
from py12box import core


py12box_path = Path(__file__).parents[1].absolute()


def strat_lifetime_tune(target_lifetime, project_path, case, species):
    """
    Tune stratospheric lifetime. Updates specified lifetime file with local lifetimes that are consistent with target
    lifetime.

    Parameters
    ----------
    target_lifetime : float
        Global stratospheric lifetime to tune local lifetimes to match
    project_path : pathlib path
        Path to project folder (e.g. py12box/example)
    case : str
        Case name (e.g. 'CFC-11_example' for py12bpx/example/CFC-11_example)
    species : str
        Species name (e.g. 'CFC-11')
    """

    # TODO: Add other lifetimes in here

    df = pd.read_csv(project_path / case / f"{species}_lifetime.csv")

    if len(df) != 12:
        raise Exception("Error: only works with annually repeating lifetimes at the moment")

    strat_invlifetime_relative = np.load(py12box_path / "inputs/strat_invlifetime_relative.npy")

    nyears = 1000

    current_lifetime = target_lifetime / 20.

    for i in range(10):
        test_lifetime = df[[f"box_{i + 1}" for i in range(12)]].values
        test_lifetime[:, 8:] = current_lifetime / strat_invlifetime_relative
        test_lifetime = np.tile(test_lifetime, (nyears, 1))

        ic = np.ones(12) * 10.
        q = np.zeros((nyears * 12, 12))
        q[:, 0] = 10.

        # Get model parameters
        mol_mass, oh_a, oh_er = startup.get_species_parameters(species)
        i_t, i_v1, t, v1, oh, cl, temperature = startup.get_model_parameters(nyears)
        F = startup.transport_matrix(i_t, i_v1, t, v1)

        c_month, burden, emissions_out, losses, lifetimes = \
            core.model(ic=ic, q=q,
                       mol_mass=mol_mass,
                       lifetime=test_lifetime,
                       F=F,
                       temp=temperature,
                       cl=cl, oh=oh,
                       arr_oh=np.array([oh_a, oh_er]))

        print(
            f"... stratosphere: {lifetimes['global_strat'][-12:].mean()}, total {lifetimes['global_total'][-12:].mean()}")

        lifetime_factor = (1. / target_lifetime) / (1. / lifetimes["global_strat"][-12:]).mean()
        current_lifetime /= lifetime_factor

    for bi in range(4):
        df[f"box_{9 + bi}"] = current_lifetime / strat_invlifetime_relative[:, bi]

    df.to_csv(project_path / case / f"{species}_lifetime.csv", index=False)


def emissions_write(time, emissions,
                    project=None,
                    case=None,
                    species=None):
    '''
    Write emissions file

    Args:
        time: N-element pandas datetime for start of each emissions time period
        emissions: 4 x N element array of emissions values in Gg
    '''

    # TODO: FINISH THIS



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

