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

12-box model class
"""

import numpy as np
from pathlib import Path
import time
from py12box import startup, core, get_data
import pandas as pd


class Model:
    #TODO: Add docstring

    # Mass of the atmosphere
    mass=5.1170001e+18 * 1000 * np.array([0.125, 0.125, 0.125, 0.125,
                                          0.075, 0.075, 0.075, 0.075,
                                          0.050, 0.050, 0.050, 0.050])

    # Hard-wiring this for now, but should be a variable
    inputs_path = get_data("inputs")

    def __init__(self, species, project_directory,
                 species_param_file=None,
                 lifetime_strat=None,
                 lifetime_ocean=None,
                 lifetime_other_trop=None):
        """Set up model class

        Parameters
        ----------
        species : str
            Species name (e.g. "CFC-11")
            Must match string in data/inputs/species_info.csv
        project_directory : pathlib.Path
            Path to project directory, which contains emissions, lifetimes, etc.
        species_param_file : str, optional
            Species parameter file. Defaults to data/inputs/species_info.csv, by default None
        """

        self.species = species
        self.project_path = project_directory

        # Get species-specific parameters
        mol_mass, oh_a, oh_er, units = startup.get_species_parameters(species,
                                                                      param_file=species_param_file)
        self.mol_mass = mol_mass
        self.oh_a = oh_a
        self.oh_er = oh_er
        self.units = units

        # Get emissions. Time is taken from emissions file
        time, emissions = startup.get_emissions(species, project_directory)
        self.time = time
        self.emissions = emissions

        # Get initial conditions
        self.ic = startup.get_initial_conditions(species, project_directory)

        # Get transport parameters, OH, Cl and temperature
        _i_t, _i_v1, _t, _v1, oh, cl, temperature = startup.get_model_parameters(len(emissions)/12,
                                                                                 input_dir=self.inputs_path)
        self.oh = oh
        self.cl = cl
        self.temperature = temperature

        # Transform transport parameters into matrix
        self.F = startup.transport_matrix(_i_t, _i_v1, _t, _v1)

        # Get lifetime
        n_years = len(time)
        #self.lifetime = startup.get_lifetime(species,
        #                                     project_directory,
        #                                     n_years)
        self.lifetime = self.tune_lifetime(lifetime_strat=lifetime_strat,
                        lifetime_trop=lifetime_other_trop,
                        lifetime_ocean=lifetime_ocean)
        
        # Run model for one timestep to compile
        print("Compiling model...")
        self.run(nsteps=1)


    def tune_lifetime(self,
                      lifetime_strat=None,
                      lifetime_trop=None,
                      lifetime_ocean=None,
                      lifetime_relative_strat_file=get_data("inputs/strat_invlifetime_relative.npy")):
        
        # Get relative stratospheric lifetime
        invlifetime_relative_strat = np.load(lifetime_relative_strat_file)
        #TODO: Add others in

        if lifetime_strat is None:
            # Get lifetime from species_info.csv
            # TODO: This is potentially dangerous. We allow a different species_info file in get_species_parameters
            # Perhaps store the parameter file string in model class, or store the whole row from that file when it is read
            df = pd.read_csv(get_data("inputs/species_info.csv"),
                            index_col="Species")
            lifetime_strat = df["Lifetime stratosphere"][self.species]

        # Start with an initial guess of the stratospheric lifetime
        current_lifetime_strat = lifetime_strat/20.

        tune_years = 100
        tune_steps = 10

        for i in range(tune_steps):
            test_lifetime = np.ones((12, 12))*1e-12
            test_lifetime[:, 8:] = current_lifetime_strat / invlifetime_relative_strat
            test_lifetime = np.tile(test_lifetime, (tune_years, 1))

            q = np.zeros((tune_years * 12, 12))
            test_emissions = np.tile(self.emissions[:12,:], (tune_years, 1))

            mole_fraction_out, burden_out, q_out, losses, global_lifetimes = \
                core.model(self.ic, test_emissions, self.mol_mass, test_lifetime,
                            self.F, self.temperature, self.oh, self.cl,
                            arr_oh=np.array([self.oh_a, self.oh_er]),
                            mass=self.mass)

            lifetime_factor = (1. / lifetime_strat) / (1. / global_lifetimes["global_strat"][-12:]).mean()
            current_lifetime_strat /= lifetime_factor
            
            print("UPDATED LIFETIME VALUE HERE")

        # Update test_lifetime to reflect last tuning step:
        out_lifetime = np.ones((12, 12))*1e-12
        out_lifetime[:, 8:] = current_lifetime_strat / invlifetime_relative_strat
        
        self.lifetime = np.tile(out_lifetime, (len(self.time), 1))

    def run(self, nsteps=-1, verbose=True):
        """Run 12-box model

        Parameters
        ----------
        nsteps : int, optional
            Number of timesteps. Ignored if set to a negative value, by default -1
        verbose : bool, optional
            Toggle verbose output, by default True
        """

        tic = time.time()

        mole_fraction_out, burden_out, q_out, losses, global_lifetimes = \
            core.model(self.ic, self.emissions, self.mol_mass, self.lifetime,
                        self.F, self.temperature, self.oh, self.cl,
                        arr_oh=np.array([self.oh_a, self.oh_er]),
                        mass=self.mass,
                        nsteps=nsteps)
        
        toc = time.time()

        if verbose:
            print(f"... done in {toc - tic} s")

        self.mf = mole_fraction_out
        self.burden = burden_out
        self.lifetimes = global_lifetimes
        self.losses = losses
        self.emissions_model = q_out


