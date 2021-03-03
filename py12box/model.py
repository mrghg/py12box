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
        # TODO: make this have a smaller number of outputs!
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

        # Get strat lifetime
        print("Tuning lifetime...")
        if lifetime_strat == None:
            lifetime_strat_tune = startup.get_species_lifetime(species, "strat")
        else:
            lifetime_strat_tune = lifetime_strat

        # Get ocean lifetime
        if lifetime_ocean == None:
            lifetime_ocean_tune = startup.get_species_lifetime(species, "ocean")
        else:
            lifetime_ocean_tune = lifetime_ocean

        self.tune_lifetime(lifetime_strat_tune,
                           lifetime_ocean_tune)

        # Run model for one timestep to compile
        #print("Compiling model...")
        #self.run(nsteps=1)


    def tune_lifetime(self,
                      lifetime_strat,
                      lifetime_ocean,
#                      lifetime_trop=None,
                      lifetime_relative_strat_file=get_data("inputs/invlifetime_relative_strat.npy"),
                      threshold=1e4,
                      tune_years=100):
        
        def local_lifetimes(global_lifetime_strat,
                            global_lifetime_ocean,
                            n_years
                            ):
            # Construct local lifetimes array from global lifetimes

            lifetime_array = np.ones((12, 12))*1e12
            if global_lifetime_strat < threshold:
                lifetime_array[:, 8:] = global_lifetime_strat / invlifetime_relative_strat
            if global_lifetime_ocean < threshold:
                lifetime_array[:, 0:4] = np.repeat([global_lifetime_ocean / invlifetime_relative_ocean],
                                                    12, axis=0)
            lifetime_array = np.tile(lifetime_array, (n_years, 1))

            return lifetime_array

        def run_lifetimes(test_lifetime):
            mole_fraction_out, burden_out, q_out, losses, global_lifetimes = \
                            core.model(self.ic, test_emissions, self.mol_mass, test_lifetime,
                                        test_f, test_temp, test_oh, test_cl,
                                        arr_oh=np.array([self.oh_a, self.oh_er]),
                                        mass=self.mass)
            return global_lifetimes

        if (lifetime_strat > threshold) * (lifetime_ocean > threshold):
            self.lifetime = local_lifetimes(lifetime_strat, lifetime_ocean, int(len(self.time)/12))
            return

        # Get relative stratospheric lifetime
        invlifetime_relative_strat = np.load(lifetime_relative_strat_file)

        # Impose relative ocean loss
        invlifetime_relative_ocean = np.array([0.18349867, 0.154188985, 0.287281601, 0.375030744])

        # Start with an initial guess of the stratospheric lifetime
        if lifetime_strat < threshold:
            current_lifetime_strat = lifetime_strat/20.
        else:
            current_lifetime_strat = lifetime_strat

        if lifetime_ocean < threshold:
            current_lifetime_ocean = lifetime_ocean/10.
        else:
            current_lifetime_ocean = lifetime_ocean

        # start this off at some value bigger than the tolerance of the while loop
        current_global_lossrate_strat = 1.1/lifetime_strat
        current_global_lossrate_ocean = 1.1/lifetime_ocean

        # Set up some arrays to run the model
        # Using mean emissions and first year of OH, Cl, temperature and F
        test_emissions = np.repeat([self.emissions.mean(axis=0)], tune_years*12, axis=0)
        test_oh = np.tile(self.oh[:12,:], (tune_years, 1))
        test_cl = np.tile(self.cl[:12,:], (tune_years, 1))
        test_temp = np.tile(self.temperature[:12,:], (tune_years, 1))
        test_f = np.tile(self.F[:12,:, :], (tune_years, 1, 1))

        # Keep track of number of iterations, to prevent infinite loop
        counter = 0

        while not (np.isclose(current_global_lossrate_strat, 1./lifetime_strat, rtol=0.01) and \
                    np.isclose(current_global_lossrate_ocean, 1./lifetime_ocean, rtol=0.01)):

            test_lifetime = local_lifetimes(current_lifetime_strat,
                                            current_lifetime_ocean,
                                            tune_years)

            global_lifetimes = run_lifetimes(test_lifetime)

            # Update lifetimes
            if lifetime_strat < threshold:
                current_global_lossrate_strat = (1./global_lifetimes["global_strat"][-12:]).mean()
                lifetime_factor_strat = (1. / lifetime_strat) / current_global_lossrate_strat
                current_lifetime_strat /= lifetime_factor_strat
            else:
                # Satisfy criterion to exit while loop
                current_global_lossrate_strat = 1./lifetime_strat

            if lifetime_ocean < threshold:
                current_global_lossrate_ocean = (1./global_lifetimes["global_othertroplower"][-12:]).mean()
                lifetime_factor_ocean = (1. / lifetime_ocean) / current_global_lossrate_ocean
                current_lifetime_ocean /= lifetime_factor_ocean
            else:
                # Satisfy criterion to exit while loop
                current_global_lossrate_ocean = 1./lifetime_ocean

            counter +=1

            if counter == 10:
                print("Exiting: lifetime didn't converge")
                break

        # Update test_lifetime to reflect last tuning step:
        self.lifetime = local_lifetimes(current_lifetime_strat,
                                        current_lifetime_ocean,
                                        int(len(self.time)/12))

        global_lifetimes = run_lifetimes(local_lifetimes(current_lifetime_strat, current_lifetime_ocean,
                                                         tune_years))

        self.steady_state_lifetime_strat = global_lifetimes['global_strat'][-12:].mean()
        self.steady_state_lifetime_ocean = global_lifetimes['global_othertroplower'][-12:].mean()

        if lifetime_strat > threshold:
            print(f"... stratospheric lifetime: 1e12")
        else:
            print(f"... stratospheric lifetime: {self.steady_state_lifetime_strat:.1f}")

        if lifetime_ocean > threshold:
            print("... ocean lifetime: 1e12")
        else:
            print(f"... ocean lifetime: {self.steady_state_lifetime_ocean:.1f}")


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

        #TODO: Tidier way needed to differentiate input and output lifetimes

        self.mf = mole_fraction_out
        self.burden = burden_out
        self.instantaneous_lifetimes = global_lifetimes
        self.losses = losses
        self.emissions_model = q_out


if __name__ == "__main__":

    mod = Model("CFC-11", get_data("example/CFC-11"))
    #tic = time.time()
    #mod.tune_lifetime(lifetime_strat=57.)
    #toc = time.time()
    #print(f"... done in {toc - tic} s")
    a=1