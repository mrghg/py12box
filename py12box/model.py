import numpy as np
from pathlib import Path
import time
from py12box import startup, core, get_data
import pandas as pd


class Model:
    """AGAGE 12-box model class
    
    This class contains inputs and outputs of the 12-box model, 
    for a particular species, emissions, initial conditions, etc.

    Attributes
    ----------
    mass : ndarray
        1d, 12
        Mass of atmosphere in g in each box
    inputs_path : pathlib.Path
        Path to the model parameter input directory
    
    """

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
                 lifetime_trop=None):
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

        print("Compiling model and tuning lifetime...")

        # Get strat lifetime        
        if lifetime_strat == None:
            lifetime_strat_tune = startup.get_species_lifetime(species, "strat")
        else:
            lifetime_strat_tune = lifetime_strat

        # Get ocean lifetime
        if lifetime_ocean == None:
            lifetime_ocean_tune = startup.get_species_lifetime(species, "ocean")
        else:
            lifetime_ocean_tune = lifetime_ocean

        # Get tropospheric lifetime
        if lifetime_trop == None:
            lifetime_trop_tune = startup.get_species_lifetime(species, "trop")
        else:
            lifetime_trop_tune = lifetime_trop

        self.tune_lifetime(lifetime_strat_tune,
                           lifetime_ocean_tune,
                           lifetime_trop_tune)

        # Run the model again with the default inputs, for 1 step, to recompile
        self.run(nsteps=1)


    def tune_lifetime(self,
                      lifetime_strat,
                      lifetime_ocean,
                      lifetime_trop,
                      lifetime_relative_strat_file=get_data("inputs/invlifetime_relative_strat.npy"),
                      threshold=1e4,
                      tune_years=100):
        """Tune the local non-OH lifetimes to a set of given global values

        Parameters
        ----------
        lifetime_strat : float
            Target global steady state stratospheric lifetime (years)
        lifetime_ocean : float
            Target global steady state lifetime with respect to ocean uptake (years)
        lifetime_trop : float
            Target global steady state lifetime with respect to non-OH tropospheric loss (years)
        lifetime_relative_strat_file : pathlib, optional
            File containing monthly relative loss rates in stratospheric boxes, 
            by default get_data("inputs/invlifetime_relative_strat.npy")
        threshold : float, optional
            Above this threshold, lifetimes are ignored, and negligible loss is asssumed (years), by default 1e4
        tune_years : int, optional
            Number of years assumed to spin the model up to steady state, by default 100

        Raises
        -----------
        Exception
            If an ocean and tropospheric lifetime are both specified. This isn't implemented yet

        """
        
        def local_lifetimes(lifetime_dict,
                            n_years
                            ):
            # Construct local lifetimes array from global lifetimes/lossrates

            lossrate_array = np.ones((12, 12))*1e-12

            if lifetime_dict["strat"]["target_global_lossrate"] > 1./threshold:
                lossrate_array[:, 8:] += lifetime_dict["strat"]["current_lossrate"] * lifetime_dict["strat"]["lossrate_relative"]
            if lifetime_dict["ocean"]["target_global_lossrate"] > 1./threshold:
                lossrate_array[:, 0:4] += lifetime_dict["ocean"]["current_lossrate"] * lifetime_dict["ocean"]["lossrate_relative"]
            if lifetime_dict["trop"]["target_global_lossrate"] > 1./threshold:
                lossrate_array[:, 0:8] += lifetime_dict["trop"]["current_lossrate"] * lifetime_dict["trop"]["lossrate_relative"]

            return np.tile(1./lossrate_array, (n_years, 1))

        def run_lifetimes(test_lifetime):
            # Run model and extract global lifetimes

            mole_fraction_out, burden_out, q_out, losses, global_lifetimes = \
                            core.model(self.ic, test_emissions, self.mol_mass, test_lifetime,
                                        test_f, test_temp, test_oh, test_cl,
                                        arr_oh=np.array([self.oh_a, self.oh_er]),
                                        mass=self.mass)
            return global_lifetimes

        def lifetimes_close(lifetime_dict, rtol=0.01):
            # Are lifetimes all close to the target?

            cl = 0
            for _, sink_data in lifetime_dict.items():
                cl += int(np.isclose(sink_data["current_global_lossrate"], sink_data["target_global_lossrate"], rtol=rtol))
            if cl == len(lifetime_dict.keys()):
                return True
            else:
                return False

        if (lifetime_ocean < threshold) and (lifetime_trop) < threshold:
            raise Exception("Not yet implemented: you can't have a finite ocean and non-OH tropospheric loss")

        lifetime_data = {"strat": {"lossrate_relative": np.load(lifetime_relative_strat_file),
                                    "current_lossrate": [1./(lifetime_strat/20.), 1./lifetime_strat][int(lifetime_strat > threshold)], #if above threshold, don't scale
                                    "current_global_lossrate": 1.1/lifetime_strat,
                                    "target_global_lossrate": 1./lifetime_strat
                                    },
                         "ocean": {"lossrate_relative": np.repeat([np.array([0.18349867, 0.154188985, 0.287281601, 0.375030744])], 12, axis=0),
                                    "current_lossrate": [1./(lifetime_ocean/10.), 1./lifetime_ocean][int(lifetime_ocean > threshold)],
                                    "current_global_lossrate": 1.1/lifetime_ocean,
                                    "target_global_lossrate": 1./lifetime_ocean
                                    },
                         "trop": {"lossrate_relative": self.oh[:12,:8]/(self.oh[:12,:8]).sum(),
                                    "current_lossrate": [1./(lifetime_trop/10.), 1./lifetime_trop][int(lifetime_trop > threshold)],
                                    "current_global_lossrate": 1.1/lifetime_trop,
                                    "target_global_lossrate": 1./lifetime_trop
                                    }
                        }

        if sum([sink_data["target_global_lossrate"] for _, sink_data in lifetime_data.items()]) < 3./threshold:

            print("... lifetimes all very large, assuming no loss")
            self.lifetime = np.tile(np.ones((12, 12))*1e12, (int(len(self.time)/12), 1))

        else:

            # Set up some arrays to run the model
            # Using mean emissions and first year of OH, Cl, temperature and F
            test_emissions = np.repeat([self.emissions.mean(axis=0)], tune_years*12, axis=0)
            test_oh = np.tile(self.oh[:12,:], (tune_years, 1))
            test_cl = np.tile(self.cl[:12,:], (tune_years, 1))
            test_temp = np.tile(self.temperature[:12,:], (tune_years, 1))
            test_f = np.tile(self.F[:12,:, :], (tune_years, 1, 1))

            # Keep track of number of iterations, to prevent infinite loop
            counter = 0

            while not lifetimes_close(lifetime_data):

                test_lifetime = local_lifetimes(lifetime_data, tune_years)
                global_lifetimes = run_lifetimes(test_lifetime)

                # Update lifetimes
                sinkstr = {"strat": "strat",
                            "ocean": "othertroplower",
                            "trop": "othertrop"}
                
                for sink, sink_data in lifetime_data.items():
                    if sink_data["target_global_lossrate"] > 1./threshold:
                        sink_data["current_global_lossrate"] = (1./global_lifetimes["global_" + sinkstr[sink]][-12:]).mean()
                        lossrate_factor = sink_data["target_global_lossrate"]/sink_data["current_global_lossrate"]
                        sink_data["current_lossrate"] *= lossrate_factor
                    else:
                        # Satisfy criterion to exit while loop
                        sink_data["current_global_lossrate"] = sink_data["target_global_lossrate"]

                counter +=1

                if counter == 50:
                    print("Exiting: lifetime didn't converge")
                    break

            print(f"... completed in {counter} iterations")

            # Update test_lifetime to reflect last tuning step:
            self.lifetime = local_lifetimes(lifetime_data,
                                            int(len(self.time)/12))

        global_lifetimes = run_lifetimes(local_lifetimes(lifetime_data,
                                                         tune_years))

        self.steady_state_lifetime_strat = global_lifetimes['global_strat'][-12:].mean()
        self.steady_state_lifetime_ocean = global_lifetimes['global_othertroplower'][-12:].mean()
        self.steady_state_lifetime_oh = global_lifetimes['global_oh'][-12:].mean()
        self.steady_state_lifetime_cl = global_lifetimes['global_cl'][-12:].mean()
        self.steady_state_lifetime_othertrop = global_lifetimes['global_othertrop'][-12:].mean()
        self.steady_state_lifetime = global_lifetimes['global_total'][-12:].mean()

        if lifetime_strat > threshold:
            print(f"... stratospheric lifetime: 1e12")
        else:
            print(f"... stratospheric lifetime: {self.steady_state_lifetime_strat:.1f}")

        if self.steady_state_lifetime_oh > threshold:
            print(f"... OH lifetime: 1e12")
        else:
            print(f"... OH lifetime: {self.steady_state_lifetime_oh:.1f}")

        if lifetime_ocean > threshold:
            print("... ocean lifetime: 1e12")
        else:
            print(f"... ocean lifetime: {self.steady_state_lifetime_ocean:.1f}")

        if self.steady_state_lifetime_othertrop > threshold:
            print(f"... non-OH tropospheric lifetime: 1e12")
        else:
            print(f"... non-OH tropospheric lifetime: {self.steady_state_lifetime_othertrop:.1f}")

        if self.steady_state_lifetime > threshold:
            print("... overall lifetime: 1e12")
        else:
            print(f"... overall lifetime: {self.steady_state_lifetime:.1f}")


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
        self.instantaneous_lifetimes = global_lifetimes
        self.losses = losses
        self.emissions_model = q_out
