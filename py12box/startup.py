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


def get_lifetime(species, project_directory, n_years):
    #TODO: have this be calculated online, removing the need for a lifetime file

    # Get lifetime
    if not os.path.isfile(project_directory / f"{species}_lifetime.csv"):
        print("No lifetime file. \n Estimating stratospheric lifetime.")
        strat_lifetime_tune(project_directory, species)
    lifetime_df = pd.read_csv(project_directory / species / f"{species}_lifetime.csv",
                              header=0, index_col=0,
                              comment="#")

    lifetime = np.tile(lifetime_df.values, (n_years, 1))

    return lifetime


def get_initial_conditions(species, project_directory):
    #TODO: docstring

    # Get initial conditions
    if not os.path.isfile(project_directory / f"{species}_initial_conditions.csv"):
        print("No inital conditions file. \n Assuming zero initial conditions")
        ic = (zero_initial_conditions().values.astype(np.float64)).flatten()
    else:
        ic = (pd.read_csv(project_directory / species / f"{species}_initial_conditions.csv",
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

def strat_lifetime_tune(project_path, species, target_lifetime=None):
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
    if not target_lifetime:
        ltdf = pd.read_csv(py12box_path / "inputs/lifetimes.csv", comment="#", index_col=0)
        target_lifetime = ltdf.loc[species][0]

    if not os.path.isfile(project_path / f"{species}_lifetime.csv"):    
        ltdict = {"month":np.arange(12).astype(int)+1}
        for i in range(1,13):
            ltdict["box_"+str(i)] = np.ones(12)*1e12 if i < 9 else np.ones(12)*10
        df = pd.DataFrame(ltdict)
        df.to_csv(project_path / f"{species}_lifetime.csv", index=False)

    df = pd.read_csv(project_path / f"{species}_lifetime.csv")
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
        mod = model.Model(species, project_path)
        mod.run(verbose=False)

        print(
            f"... stratosphere: {mod.lifetimes['global_strat'][-12:].mean()}, total {mod.lifetimes['global_total'][-12:].mean()}")

        lifetime_factor = (1. / target_lifetime) / (1. / mod.lifetimes["global_strat"][-12:]).mean()
        current_lifetime /= lifetime_factor

    for bi in range(4):
        df[f"box_{9 + bi}"] = current_lifetime / strat_invlifetime_relative[:, bi]

    df.to_csv(project_path / f"{species}_lifetime.csv", index=False)



