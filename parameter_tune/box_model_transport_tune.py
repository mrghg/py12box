#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:15:10 2019

@author: chxmr
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from py12box import setup, core
import matplotlib as mpl
import time as systime

mpl.style.use('ggplot')

py12box_path = Path(__file__).parents[1] / "py12box"
py12box_projects_path = Path(__file__).parents[1] / "py12box_projects/"

with open("/home/chxmr/work/py12box_parameters/co2_boxed_up.p", "rb") as f:
    box_data = pickle.load(f)
    
mf_mozart = box_data["mf"]
q_mozart = box_data["q"]

df = pd.DataFrame(index = np.linspace(2015., 2016. - 1./12., num = 12),
                  data = q_mozart.T/1e6,
                  columns = ["box_1", "box_2", "box_3", "box_4"])

df.to_csv(py12box_projects_path / "CO2_emissions.csv")

dir_path = Path(__file__).parent
input_dir = py12box_path / "inputs"
project_dir = py12box_projects_path / "parameter_tune"

species_info = pd.read_csv(input_dir / "species_info.csv",
                           index_col = "Species")

case = "CO2_2015"
species = "CO2"
mol_mass = species_info["Molecular mass (g/mol)"][species]

time, emissions, ic, lifetime = setup.get_species_parameters(project_dir,
                                                             case,
                                                             species)

i_t, i_v1, t, v1, OH, Cl, temperature = setup.get_model_parameters(input_dir,
                                                                   int(len(time) / 12))

def run_model(theta):

    t = np.repeat(theta[0:nt*4].reshape((4, nt)), 3, axis = 0)
    v1 = np.repeat(theta[nt*4:].reshape((4, nv1)), 3, axis = 0)

    F = setup.transport_matrix(i_t, i_v1, t, v1)
    c_month, burden, emissions_out, losses, lifetimes = \
        core.model(ic=ic, q=emissions,
                   mol_mass=mol_mass,
                   lifetime=lifetime,
                   F=F,
                   temp=temperature,
                   Cl=Cl, OH=OH)

    return(c_month)


nv1 = len(v1[0, :])
nt = len(t[0, :])

si = range(0, 12, 3)
theta_t = t[si, :].flatten()
theta_v1 = v1[si, :].flatten()
theta = np.hstack([theta_t, theta_v1])
dtheta = theta*0.2

c_month = run_model(theta)


nIt = 100

print("Running...")
start = systime.time()

c_month_pert = np.zeros((12, 12, nIt))
for it in range(nIt):
    c_month_pert[:, :, it] = run_model(theta + dtheta * np.random.randn((nv1 + nt)*4))

end = systime.time()
print(" ... %s s" % (end - start))

c_month_std = np.std(c_month_pert, axis = 2)


box = 0
for (box, title) in zip([0, 4, 8],
                        ["Boundary layer", "Troposphere", "Stratosphere"]):
    plt.fill_between(time,
                     (c_month[:, box] - c_month_std[:, box])/1e6,
                     (c_month[:, box] + c_month_std[:, box])/1e6,
                     color = "blue",
                     alpha = 0.5)
    plt.plot(time, c_month[:, box]/1e6, color = "blue")
    
    plt.fill_between(time,
                     (c_month[:, box+3] - c_month_std[:, box+3])/1e6,
                     (c_month[:, box+3] + c_month_std[:, box+3])/1e6,
                     color = "orange",
                     alpha = 0.5)
    plt.plot(time, c_month[:, box+3]/1e6, color = "orange")
    plt.ylabel("%s (pmol mol$^{-1}$)" % species)
    plt.title(title)
    
    plt.plot(time, mf_mozart[box, :]*1e6, ".", color = "blue")
    plt.plot(time, mf_mozart[box+3, :]*1e6, ".", color = "orange")
    
    plt.show()


