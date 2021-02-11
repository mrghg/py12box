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

import py12box.core as core
import py12box.util as util


py12box_path = Path(__file__).parents[1].absolute()


def get_species_parameters(species,
                           param_file=None):
    """

    Parameters
    ----------
    species : str

    Returns
    ---------
    tuple containing: (molecular mass (g/mol), OH Arrhenius A, OH Arrhenius E/R

    """

    if param_file == None:
        param_file_str = "species_info.csv"
    else:
        param_file_str = param_file

    df = pd.read_csv(py12box_path / "data/inputs" / param_file_str,
                     index_col="Species")

    unit_strings = {"ppm": 1e-6,
                    "ppb": 1e-9,
                    "ppt": 1e-12,
                    "ppq": 1e-15}

    return df["Molecular mass (g/mol)"][species], \
            df["OH_A"][species], \
            df["OH_ER"][species], \
            unit_strings[df["Unit"][species]]


def get_emissions(species, project_directory):
    #TODO: Split out emissions and lifetimes
    #TODO: Add docstring

    # Get species-specfic parameters
    ####################################################

    # Get emissions
    emissions_df = pd.read_csv(project_directory / species / f"{species}_emissions.csv",
                               header=0, index_col=0,
                               comment="#")
    time_in = emissions_df.index.values

    # Work out time frequency and interpolate, if required
    time_freq = time_in[1] - time_in[0]
    if time_freq == 1:
        # Annual emissions. Interpolate to monthly
        time = np.arange(time_in[0], time_in[-1] + 1 - 1. / 12, 1 / 12.)
        emissions = np.repeat(emissions_df.values, 12, axis=0)
    else:
        # Assume monthly emissions
        time = time_in.copy()
        emissions = emissions_df.values

    return time, emissions


def get_lifetime(species, project_directory, n_years):
    #TODO: have this be calculated online, removing the need for a lifetime file

    # Get lifetime
    lifetime_df = pd.read_csv(project_directory / species / f"{species}_lifetime.csv",
                              header=0, index_col=0,
                              comment="#")
    lifetime = np.tile(lifetime_df.values, (n_years, 1))

    return lifetime


def get_initial_conditions(species, project_directory):
    #TODO: docstring

    # Get initial conditions
    ic = (pd.read_csv(project_directory / species / f"{species}_initial_conditions.csv",
                      header=0,
                      comment="#").values.astype(np.float64)).flatten()

    return ic


def get_model_parameters(n_years, input_dir=py12box_path / "data/inputs"):
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
