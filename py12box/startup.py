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
    float
        Molecular mass
    float
        OH "A" Arrhenuis parameter
    float
        OH "E/R" Arrhenius parameter
    float
        unit (e.g. 1e-12 for ppt)

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


def get_species_lifetime(species,
                         which_lifetime,
                         param_file=None):
    """Get lifetimes for a specific species

    Parameters
    ----------
    species : str
        Species name. Must match species_info.csv
    which_lifetime : str
        Either "strat", "ocean" or "trop"
    param_file : str, optional
        Name of species info file, by default None, which sets species_info.csv

    Returns
    ----------
    float
        Lifetime value

    Raises
    ----------
    Exception
        If which_lifetime is not a valid input
    
    """

    if param_file == None:
        param_file_str = "species_info.csv"
    else:
        #TODO: Put this outside the main package
        param_file_str = param_file

    df = pd.read_csv(get_data("inputs") / param_file_str,
                     index_col="Species")

    if which_lifetime == "strat":
        out_lifetime = df["Lifetime stratosphere (years)"][species]
    elif which_lifetime == "ocean":
        out_lifetime = df["Lifetime ocean (years)"][species]
    elif which_lifetime == "trop":
        out_lifetime = df["Lifetime other troposphere (years)"][species]
    else:
        raise Exception("Not a valid input to which_lifetime")

    if not np.isfinite(out_lifetime):
        return 1e12
    else:
        return out_lifetime


def zero_initial_conditions():
    """
    Make an initial conditions dataframe with all boxes 1e-12

    Returns
    -------
    pandas.DataFrame
        Dataframe of initial conditions equal to 1e-12

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
    ndarray
        1d, ntimesteps
        Array containing decimal times (1 x ntimesteps)
    ndarray
        2d, 12 x ntimesteps
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


def get_initial_conditions(species, project_directory):
    """Read initial conditions from file

    Parameters
    ----------
    species : str
        Species name
    project_directory : pathlib.Path
        Path to project folder, containing <species>_initial_conditions.csv file

    Returns
    -------
    ndarray
        1d, 12
        Initial conditions in each box
    
    """

    if not (project_directory / f"{species}_initial_conditions.csv").exists():
        print("No inital conditions file. \n Assuming zero initial conditions")
        ic = (zero_initial_conditions().values.astype(np.float64)).flatten()
    else:
        ic = (pd.read_csv(project_directory / f"{species}_initial_conditions.csv",
                          header=0,
                          comment="#").values.astype(np.float64)).flatten()
    return ic


def get_model_parameters(n_years, input_dir=get_data("inputs")):
    """Retrieve model transport parameters, OH, Cl and temperature and repeat annually

    Parameters
    ----------
    n_years : int
        Number of years over which to repeat parameters
    input_dir : pathlib.Path, optional
        Directory containing parameter files, by default get_data("inputs")

    Returns
    -------
    ndarray
        3d, n_years, n_box_intersections_diffusion x 2
        Intersections to apply to the mixing timescales for staggered grids.
    ndarray
        3d, n_years, n_box_intersections_advection x 2
        Intersections to apply to the velocity timescales for staggered grids
        excluding stratosphere.
    ndarray
        2d, n_years, n_box_intersections_diffusion
        Mixing timescales for staggered grids (s).
    ndarray
        2d, n_years, n_box_intersections_diffusion_advection
        Velocity timescales for staggered grids excluding stratosphere (s).
    ndarray
        2d, n_years*12, 12
        OH concentration in each box for each month
    ndarray
        2d, n_years*12, 12
        Cl concentration in each box for each month
    ndarray
        2d, n_years*12, 12
        Temperature in each box for each month

    """

    # Get transport parameters and create transport matrix
    i_t, i_v1, t, v1 = \
        util.io_r_npz(os.path.join(input_dir,
                                   'transport.npz'))
    t = np.tile(t, (int(n_years), 1))
    v1 = np.tile(v1, (int(n_years), 1))

    # Get OH
    oh = np.tile(util.io_r_npy(os.path.join(input_dir, 'OH.npy')),
                 (int(n_years), 1))

    # Get Cl
    cl = np.tile(util.io_r_npy(os.path.join(input_dir, 'Cl.npy')),
                 (int(n_years), 1))

    # Get temperature
    temperature = np.tile(util.io_r_npy(os.path.join(input_dir,
                                                     'temperature.npy')),
                          (int(n_years), 1))

    return i_t, i_v1, t, v1, oh, cl, temperature


def transport_matrix(i_t, i_v1, t, v1):
    """Construct transport matrix from transport parameters

    Parameters
    ----------
    i_t : ndarray
        3d, n_years x n_box_intersections_diffusion x 2
        Intersections to apply to the mixing timescales for staggered grids.
    i_v1 : ndarray
        3d, n_years x n_box_intersections_advection x 2
        Intersections to apply to the velocity timescales for staggered grids
        excluding stratosphere.
    t : ndarray
        2d, n_years x n_box_intersections_diffusion
        Mixing timescales for staggered grids (s).
    v1 : ndarray
        2d, n_years x n_box_intersections_diffusion_advection
        Velocity timescales for staggered grids excluding stratosphere (s).

    Returns
    -------
    ndarray
        n_months x 12 x 12
        Transport matrix
    
    """

    n_months = t.shape[0]
    t *= (24.0 * 3600.0)
    v1 *= (24.0 * 3600.0)
    F = np.zeros((n_months, 12, 12))
    for mi in range(0, n_months):
        F[mi] = core.model_transport_matrix(i_t=i_t, i_v1=i_v1,
                                            t_in=t[mi],
                                            v1_in=v1[mi])
    return F


