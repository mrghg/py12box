#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:05:03 2018

@author: chxmr
"""

import numpy as np
import py12box.core as core
import py12box.util as util
import os
import pandas as pd
from pathlib import Path


py12box_path = Path(__file__).parent.absolute()


def get_species_parameters(species):
    """

    Parameters
    ----------
    species : str

    Returns
    ---------
    tuple containing: (molecular mass (g/mol), OH Arrhenius A, OH Arrhenius E/R

    """
    df = pd.read_csv(py12box_path / "inputs/species_info.csv",
                     index_col="Species")

    return df["Molecular mass (g/mol)"][species], df["OH_A"][species], df["OH_ER"][species]


def get_case_parameters(project_directory,
                        case,
                        species):
    # Get species-specfic parameters
    ####################################################

    case_dir = project_directory / case

    # Get emissions
    emissions_df = pd.read_csv(case_dir / ("%s_emissions.csv" % species),
                               header=0, index_col=0)
    time_in = emissions_df.index.values

    # Get time from emissions file
    n_years = len(time_in)

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

    # Get lifetime
    lifetime_df = pd.read_csv(case_dir / ('%s_lifetime.csv' % species),
                              header=0, index_col=0)
    lifetime = np.tile(lifetime_df.values, (n_years, 1))

    # Get initial conditions
    ic = (pd.read_csv(case_dir / ('%s_initial_conditions.csv' % species),
                      header=0).values.astype(np.float64)).flatten()

    return time, emissions, ic, lifetime


def get_model_parameters(n_years, input_dir=py12box_path / "inputs"):
    # Get model parameters
    ###################################################

    # Get transport parameters and create transport matrix
    i_t, i_v1, t, v1 = \
        util.io_r_npz(os.path.join(input_dir,
                                   'transport.npz'))
    t = np.tile(t, (n_years, 1))
    v1 = np.tile(v1, (n_years, 1))

    # Get OH
    OH = np.tile(util.io_r_npy(os.path.join(input_dir, 'OH.npy')),
                 (n_years, 1))

    # Get Cl
    Cl = np.tile(util.io_r_npy(os.path.join(input_dir, 'Cl.npy')),
                 (n_years, 1))

    # Get temperature
    temperature = np.tile(util.io_r_npy(os.path.join(input_dir,
                                                     'temperature.npy')),
                          (n_years, 1))

    return i_t, i_v1, t, v1, OH, Cl, temperature


def transport_matrix(i_t, i_v1, t, v1):
    n_months = t.shape[0]
    t *= (24.0 * 3600.0)
    v1 *= (24.0 * 3600.0)
    F = np.zeros((n_months, 12, 12))
    for mi in range(0, n_months):
        F[mi] = core.model_transport_matrix(i_t=i_t, i_v1=i_v1,
                                           t_in=t[mi],
                                           v1_in=v1[mi])
    return F


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
        mol_mass, oh_a, oh_er = get_species_parameters(species)
        i_t, i_v1, t, v1, oh, cl, temperature = get_model_parameters(nyears)
        F = transport_matrix(i_t, i_v1, t, v1)

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



