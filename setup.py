#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:05:03 2018

@author: chxmr
"""

import numpy as np
import core
import util
import os
import pandas as pd
from pathlib import Path
from pyprojroot import here

def get_species_parameters(species):
    
    df = pd.read_csv(here() / "inputs/species_info.csv",
                     index_col = "Species")

    return(df["Molecular mass (g/mol)"][species],
           df["OH_A"][species],
           df["OH_ER"][species])
    

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
        emissions = np.repeat(emissions_df.values, 12, axis = 0)
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

    return (time, emissions, ic, lifetime)


def get_model_parameters(input_dir, n_years):
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

    return (i_t, i_v1, t, v1, OH, Cl, temperature)


def transport_matrix(i_t, i_v1, t, v1):
    
    n_months = t.shape[0]
    t *= (24.0 * 3600.0)
    v1 *= (24.0 * 3600.0)
    F = np.zeros((n_months, 12, 12))
    for mi in range(0, n_months):
        F[mi] = core.model_transport_matrix(i_t=i_t, i_v1=i_v1,
                                            t_in=t[mi],
                                            v1_in=v1[mi])
    return (F)

