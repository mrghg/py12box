import numpy as np
from pathlib import Path
import time
import pandas as pd
from bisect import bisect
from py12box import startup, core, get_data


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
                 lifetime_trop=None,
                 start_year=None):
        """Set up model class

        Parameters:
            species : str
                Species name (e.g. "CFC-11")
                Must match string in data/inputs/species_info.csv
            project_directory : pathlib.Path
                Path to project directory, which contains emissions, lifetimes, etc.
            species_param_file : str, optional
                Species parameter file. Defaults to data/inputs/species_info.csv, by default None
            lifetime_strat : float, optional
                Stratospheric lifetime in years, by default None
            lifetime_ocean: float, optional
                Lifetime with respect to loss to the ocean in years, by default None
            lifetime_trop : float, optional
                Lifetime with respect to non-OH tropospheric loss in years (e.g. photolysis), by default None
            start_year : flt, optional
                Optional year to start the model run. Must be after first year in emissions file.
                If specified, model will run using emissions and initial conditions value from file.
                Initial conditions will be updated to the new start year from model run.
                
        Returns:
            self : 
                returns an instance of self.
                
        Attributes:
            mol_mass : float
                Molecular mass
            oh_a :  float
                OH "A" Arrhenuis parameter
            oh_er : float
                OH "E/R" Arrhenuis parameter
            units : float
                units for mole fraction (currently all stored at 1e-12 for ppt)
            time : array
                Array containing decimal times (1 x ntimesteps)
            emissions : array
                Array containing emissions (12 x ntimesteps)
            ic : array
                Initial conditions in each box
            oh : array
                OH concentration in each box for each month
            cl : array
                Cl concentration in each box for each month
            temperature : 
                Temperature in each box for each month
            F : array
                Transport matrix
            lifetime : array
                Global lifetime in each box (years)
            steady_state_lifetime_strat :
                Global steady state stratospheric lifetime (years)
            steady_state_lifetime_ocean : 
                Global ocean steady state lifetime (years)
            steady_state_lifetime_oh : 
                Global steady state lifetime with respect OH loss (years)
            steady_state_lifetime_cl : 
                Global steady state lifetime with respect Cl loss (years)
            steady_state_lifetime_othertrop : 
                Global steady state lifetime with respect Cl loss (years)
            steady_state_lifetime :
                Global steady state lifetime (years)
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

        if start_year != None:
            self.change_start_year(start_year)


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

            mole_fraction_out, mole_fraction_restart, burden_out, q_out, losses, global_lifetimes = \
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

        # Store lifetimes
        self.steady_state_lifetime_strat = global_lifetimes['global_strat'][-12:].mean()
        self.steady_state_lifetime_ocean = global_lifetimes['global_othertroplower'][-12:].mean()
        self.steady_state_lifetime_oh = global_lifetimes['global_oh'][-12:].mean()
        self.steady_state_lifetime_cl = global_lifetimes['global_cl'][-12:].mean()
        self.steady_state_lifetime_othertrop = global_lifetimes['global_othertrop'][-12:].mean()
        self.steady_state_lifetime = global_lifetimes['global_total'][-12:].mean()

        # Print outputs
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


    def change_start_year(self, start_year):
        """Change first model year

        Parameters
        ----------
        start_year : flt
            New first year for simulation
        """
        
        if start_year > self.time[0] or np.isclose(start_year, self.time[0]):

            # If no previous model run, do one for initial conditions
            if hasattr(self, 'mf'):
                if not np.isfinite(self.mf[0, 0]):
                    self.run(verbose=False)
            else:
                self.run(verbose=False)

            # Find which timestep to start at
            ti = bisect(self.time, start_year) - 1
            # Trim input arrays
            self.time = self.time[ti:]
            self.emissions = self.emissions[ti:, :]
            self.lifetime = self.lifetime[ti:, :]
            self.temperature = self.temperature[ti:, :]
            self.oh = self.oh[ti:, :]
            self.cl = self.cl[ti:, :]
            self.F = self.F[ti:, :, :]
            # Initial conditions
            if ti > 0:
                self.ic = self.mf_restart[ti-1, :]
            self.mf = self.mf[ti:, :]
            self.mf_restart = self.mf_restart[ti:, :]
            self.burden = self.burden[ti:, :]
            self.emissions_model = self.emissions_model[ti:, :]
            for key, val in self.losses.items():
                self.losses[key] = val[ti:, :]
            for key, val in self.instantaneous_lifetimes.items():
                self.instantaneous_lifetimes[key] = val[ti:]

        else:
            raise Exception("Start year can't be before first year in emissions file")


    def change_end_year(self, end_year):
        """Change last model year

        Parameters
        ----------
        end_year : flt
            New end year for simulation. 
            Note that the simulation will be trimmed before the beginning of end_year.
            I.e. end_year=2001. will curtail the simulation at the end of December 2000.
        """
        if np.isclose(end_year, self.time[-1]) or float(end_year) < self.time[-1]:
            # Trim at new end date
            ti = bisect(self.time, float(end_year)) - 1
            self.time = self.time[:ti]
            self.emissions = self.emissions[:ti, :]
            self.lifetime = self.lifetime[:ti, :]
            self.temperature = self.temperature[:ti, :]
            self.oh = self.oh[:ti, :]
            self.cl = self.cl[:ti, :]
            self.F = self.F[:ti, :, :]

            # If a model run has previously been carried out, trim those outputs too
            if hasattr(self, "mf"):
                self.mf = self.mf[:ti, :]
                self.mf_restart = self.mf_restart[:ti, :]
                self.burden = self.burden[:ti, :]
                self.emissions_model = self.emissions_model[:ti, :]
                for key, val in self.losses.items():
                    self.losses[key] = val[:ti, :]
                for key, val in self.instantaneous_lifetimes.items():
                    self.instantaneous_lifetimes[key] = val[:ti]
        else:
            raise Exception("End year can't be after last year in emissions file")

    def run(self, nsteps=-1, verbose=True):
        """Run 12-box model

        Parameters:
            nsteps : int, optional
                Number of timesteps. Ignored if set to a negative value, by default -1
            verbose : bool, optional
                Toggle verbose output, by default True
                
        Returns:
            self : 
                returns an instance of self.
        
        Attributes:
            mf : array
                Monthly mean mole fractions (pmol/mol).
            mf_restart : 
                Instantaneous mole fraction at final step of each month (pmol/mol).
            burden : array
                The monthly-average global burden (g).
            instantaneous_lifetimes : dict
                Dictionary of monthly instantaneous lifetimes 
            losses : dict
                Loss in each box for each month due to OH, Cl and other
            emissions_model :
                The monthly-average mass emissions (g).       
        """

        tic = time.time()

        mole_fraction_out, mole_fraction_restart, burden_out, q_out, losses, global_lifetimes = \
            core.model(self.ic, self.emissions, self.mol_mass, self.lifetime,
                        self.F, self.temperature, self.oh, self.cl,
                        arr_oh=np.array([self.oh_a, self.oh_er]),
                        mass=self.mass,
                        nsteps=nsteps)
        
        toc = time.time()

        if verbose:
            print(f"... done in {toc - tic:.4f} s")

        self.mf = mole_fraction_out
        self.mf_restart = mole_fraction_restart
        self.burden = burden_out
        self.instantaneous_lifetimes = global_lifetimes
        self.losses = losses
        self.emissions_model = q_out
