#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:05:03 2018

@author: chxmr
"""

import numpy as np
from py12box import core, util
import os
import pandas as pd

input_dir = "/Users/chxmr/Work/Projects/py12box/inputs"
case_dir = "/Users/chxmr/Work/Projects/py12box/example"

def tile(var):
    r'''Tile input variables
    '''
    if len(var) != n_months:
        var = np.tile(var, (n_years, 1))
    return(var)


n_box = 12    
 
# Get emissions
emissions_df = pd.read_csv(os.path.join(case_dir, 'emissions.csv'), header=0, index_col=0)
time = emissions_df.index.values
emissions = emissions_df.values


# Get time from emissions file
n_years = len(emissions)
n_months = n_years*12


# Repeat emissions
emissions = np.tile(emissions, (12, 1))
#%%

# Get transport parameters and create transport matrix
i_t, i_v1, t, v1 = \
    util.io_r_npz(os.path.join(input_dir, 'transport.npz'))
t = tile(t * 24.0 * 3600.0)
v1 = tile(v1 * 24.0 * 3600.0)
F = np.zeros((n_months, n_box, n_box))
for mi in range(0, n_months):
    F[mi] = core.model_transport_matrix(i_t=i_t, i_v1=i_v1,
                                        t_in=t[mi],
                                        v1_in=v1[mi])

# Get OH
OH = tile(util.io_r_npy(os.path.join(input_dir, 'OH.npy')))

# Get Cl
Cl = tile(util.io_r_npy(os.path.join(input_dir, 'Cl.npy')))

# Get temperature
temp = tile(util.io_r_npy(os.path.join(input_dir, 'temperature.npy')))

# Get lifetime
lifetime_df = pd.read_csv(os.path.join(case_dir, 'lifetime.csv'), header = 0, index_col = 0)
lifetime = tile(lifetime_df.values)

# Get initial conditions
ic = (pd.read_csv(os.path.join(case_dir, 'initial_conditions.csv'), header = 0).values.astype(np.float64)).flatten()


c_month, burden, emissions_out, losses, lifetimes = \
    core.run_model(ic=ic, q=emissions,
                   mol_mass=100.,
                   lifetime=lifetime,
                   F=F,
                   temp=temp,
                   Cl=Cl, OH=OH)
