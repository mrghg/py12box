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

Model startup functions.
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
from py12box import core, util, get_data, model


def get_species_parameters(species,
                           param_file=None):
    """Get parameters for a specific species (e.g. mol_mass, etc.)

    Parameters
    ----------
    species : str
        Species name. Must match species_info.csv
    param_file : str, optional
        Name of species info file, by default None, which sets species_info.csv

    Returns
    -------
    [type]
        [description]
    """

    if param_file == None:
        param_file_str = "species_info.csv"
    else:
        #TODO: Put this outside the main package
        param_file_str = param_file

    df = pd.read_csv(get_data("inputs") / param_file_str,
                     index_col="Species")

    unit_strings = {"ppm": 1e-6,
                    "ppb": 1e-9,
                    "ppt": 1e-12,
                    "ppq": 1e-15}

    return df["Molecular mass (g/mol)"][species], \
            df["OH_A"][species], \
            df["OH_ER"][species], \
            unit_strings[df["Unit"][species]]


def get_species_lifetime(species,
                         which_lifetime,
                         param_file=None):
    """Get lifetimes for a specific species

    Parameters
    ----------
    species : str
        Species name. Must match species_info.csv
    which_lifetime : str
        Either "strat", "ocean" or "trop"
    param_file : str, optional
        Name of species info file, by default None, which sets species_info.csv

    Returns
    -------
    float
        Lifetime value
    """

    if param_file == None:
        param_file_str = "species_info.csv"
    else:
        #TODO: Put this outside the main package
        param_file_str = param_file

    df = pd.read_csv(get_data("inputs") / param_file_str,
                     index_col="Species")

    if which_lifetime == "strat":
        out_lifetime = df["Lifetime stratosphere"][species]
    elif which_lifetime == "ocean":
        out_lifetime = df["Lifetime ocean"][species]
    elif which_lifetime == "trop":
        out_lifetime = df["Lifetime other troposphere"][species]
    else:
        raise Exception("Not a valid input to which_lifetime")

    if not np.isfinite(out_lifetime):
        return 1e12
    else:
        return out_lifetime


def zero_initial_conditions():
    """
    Make an initial conditions files with all boxes 1e-12

    """

    icdict = {}
    for i in range(1,13):
        icdict["box_"+str(i)] = [1e-12]
    df = pd.DataFrame(icdict)
    return df
        

def get_emissions(species, project_directory):
    """Get emissions from project's emissions file

    Parameters
    ----------
    species : str
        Species name to look up emissions file in project folder
        (e.g. "CFC-11_emissions.csv")
    project_directory : pathlib.Path
        Path to 12-box model project

    Returns
    -------
    np.array
        Array containing decimal times (1 x ntimesteps)
    np.array 
        Array containing emissions (12 x ntimesteps)
    """

    # Get emissions
    if not os.path.isfile(project_directory / f"{species}_emissions.csv"):
        raise Exception("There must be an emissions file. Please make one.")

    emissions_df = pd.read_csv(project_directory / f"{species}_emissions.csv",
                               header=0, index_col=0, comment="#")

    time_in = emissions_df.index.values

    # Work out time frequency and interpolate, if required
    time_freq = time_in[1] - time_in[0]
    if time_freq == 1:
        # Annual emissions. Interpolate to monthly
        time = np.arange(time_in[0], time_in[-1] + 1, 1 / 12.)
        emissions = np.repeat(emissions_df.values, 12, axis=0)
    else:
        # Assume monthly emissions
        time = time_in.copy()
        emissions = emissions_df.values

    return time, emissions


def get_initial_conditions(species, project_directory):
    #TODO: docstring

    # Get initial conditions
    if not (project_directory / f"{species}_initial_conditions.csv").exists():
        print("No inital conditions file. \n Assuming zero initial conditions")
        ic = (zero_initial_conditions().values.astype(np.float64)).flatten()
    else:
        ic = (pd.read_csv(project_directory / f"{species}_initial_conditions.csv",
                          header=0,
                          comment="#").values.astype(np.float64)).flatten()
    return ic


def get_model_parameters(n_years, input_dir=get_data("inputs")):
    #TODO: docstring
    # Get model parameters
    ###################################################

    # Get transport parameters and create transport matrix
    i_t, i_v1, t, v1 = \
        util.io_r_npz(os.path.join(input_dir,
                                   'transport.npz'))
    t = np.tile(t, (int(n_years), 1))
    v1 = np.tile(v1, (int(n_years), 1))

    # Get OH
    OH = np.tile(util.io_r_npy(os.path.join(input_dir, 'OH.npy')),
                 (int(n_years), 1))

    # Get Cl
    Cl = np.tile(util.io_r_npy(os.path.join(input_dir, 'Cl.npy')),
                 (int(n_years), 1))

    # Get temperature
    temperature = np.tile(util.io_r_npy(os.path.join(input_dir,
                                                     'temperature.npy')),
                          (int(n_years), 1))

    return i_t, i_v1, t, v1, OH, Cl, temperature


def transport_matrix(i_t, i_v1, t, v1):
    #TODO: docstring

    n_months = t.shape[0]
    t *= (24.0 * 3600.0)
    v1 *= (24.0 * 3600.0)
    F = np.zeros((n_months, 12, 12))
    for mi in range(0, n_months):
        F[mi] = core.model_transport_matrix(i_t=i_t, i_v1=i_v1,
                                            t_in=t[mi],
                                            v1_in=v1[mi])
    return F


