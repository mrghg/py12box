from numba import jit, njit
import numpy as np


@njit()
def model_solver_rk4(chi, F, dt):
    """Vectorized Runge-Kutta

    Parameters
    ----------
    chi : ndarray
        1d
        Burden (g).
    F : ndarray
        2d, n_box
        Transport matrix.
    dt : float
        Delta t (s).

    Returns
    -------
    chi : ndarray
        1d
        Burden (g).

    """
    A = np.dot(F, chi)
    B = np.dot(F, (chi + dt * A / 2.0))
    C = np.dot(F, (chi + dt * B / 2.0))
    D = np.dot(F, (chi + dt * C))
    chi += dt / 6.0 * (A + 2.0 * (B + C) + D)
    return chi


@jit(nopython=True)
def model_transport_matrix(i_t, i_v1, t_in, v1_in):
    """Calculate transport matrix
    
    Based on equations in:
    Cunnold, D. M. et al. (1983).
    The Atmospheric Lifetime Experiment 3. Lifetime Methodology and
    Application to Three Years of CFCl3 Data.
    Journal of Geophysical Research, 88(C13), 8379-8400.

    This function outputs a 12x12 matrix (F), calculated by collecting terms in
    the full equation scheme written out in doc_F_equation.txt.
    model transport is then calculated as dc/dt=F##c.

    Parameters
    ----------
    i_t : ndarray
        2d, n_box_stag x 2
        Intersections to apply to the mixing timescales for staggered grids.
    i_v1 : ndarray
        2d, n_box_stag_es x 2
        Intersections to apply to the velocity timescales for staggered grids
        excluding stratosphere.
    t : ndarray
        1d, n_box_stag
        Mixing timescales for staggered grids (s).
    v1 : ndarray
        1d, n_box_stag_es
        Velocity timescales for staggered grids excluding stratosphere (s).

    Returns
    -------
    F : ndarray
        2d, n_box x n_box
        Transport matrix

    """
    F = np.zeros((12, 12))
    t = np.zeros((12, 12))
    v = np.zeros((12, 12))

    for i in range(0, len(i_v1)):
        v[i_v1[i, 1], i_v1[i, 0]] = 1.0 / v1_in[i]
    for i in range(0, len(i_t)):
        t[i_t[i, 1], i_t[i, 0]] = t_in[i]

    F[0, 0] = v[1, 0] / 2.0 - v[0, 4] / 2.0 - 1.0 / t[1, 0] - 1.0 / t[0, 4]
    F[0, 1] = v[1, 0] / 2.0 + 1.0 / t[1, 0]
    F[0, 4] = - v[0, 4] / 2.0 + 1.0 / t[0, 4]

    F[1, 0] = - v[1, 0] / 2.0 + 1.0 / t[1, 0]
    F[1, 1] = v[2, 1] / 2.0 - v[1, 5] / 2.0 - v[1, 0] / 2.0 - \
              1.0 / t[2, 1] - 1.0 / t[1, 5] - 1.0 / t[1, 0]
    F[1, 2] = v[2, 1] / 2.0 + 1.0 / t[2, 1]
    F[1, 5] = - v[1, 5] / 2.0 + 1.0 / t[1, 5]

    F[2, 1] = -v[2, 1] / 2.0 + 1.0 / t[2, 1]
    F[2, 2] = v[3, 2] / 2.0 - v[2, 6] / 2.0 - v[2, 1] / 2.0 - \
              1.0 / t[3, 2] - 1.0 / t[2, 6] - 1.0 / t[2, 1]
    F[2, 3] = v[3, 2] / 2.0 + 1.0 / t[3, 2]
    F[2, 6] = -1.0 * v[2, 6] / 2.0 + 1.0 / t[2, 6]

    F[3, 2] = -v[3, 2] / 2.0 + 1.0 / t[3, 2]
    F[3, 3] = -v[3, 7] / 2.0 - v[3, 2] / 2.0 - 1.0 / t[3, 7] - 1.0 / t[3, 2]
    F[3, 7] = -v[3, 7] / 2.0 + 1.0 / t[3, 7]

    F[4, 0] = 5.0 / 3.0 * v[0, 4] / 2.0 + 5.0 / 3.0 / t[0, 4]
    F[4, 4] = 5.0 / 3.0 * v[5, 4] / 2.0 + 5.0 / 3.0 * v[0, 4] / 2.0 - \
              1.0 / t[5, 4] - 5.0 / 3.0 / t[0, 4] - 1.0 / t[4, 8]
    F[4, 5] = 5.0 / 3.0 * v[5, 4] / 2.0 + 1.0 / t[5, 4]
    F[4, 8] = 1.0 / t[4, 8]

    F[5, 1] = 5.0 / 3.0 * v[1, 5] / 2.0 + 5.0 / 3.0 / t[1, 5]
    F[5, 4] = -5.0 / 3.0 * v[5, 4] / 2.0 + 1.0 / t[5, 4]
    F[5, 5] = 5.0 / 3.0 * v[6, 5] / 2.0 + 5.0 / 3.0 * v[1, 5] / 2.0 - \
              5.0 / 3.0 * v[5, 4] / 2.0 - 1.0 / t[6, 5] - \
              5.0 / 3.0 / t[1, 5] - 1.0 / t[5, 9] - 1.0 / t[5, 4]
    F[5, 6] = 5.0 / 3.0 * v[6, 5] / 2.0 + 1.0 / t[6, 5]
    F[5, 9] = 1.0 / t[5, 9]

    F[6, 2] = 5.0 / 3.0 * v[2, 6] / 2.0 + 5.0 / 3.0 / t[2, 6]
    F[6, 5] = -5.0 / 3.0 * v[6, 5] / 2.0 + 1.0 / t[6, 5]
    F[6, 6] = 5.0 / 3.0 * v[7, 6] / 2.0 + 5.0 / 3.0 * v[2, 6] / 2.0 - \
              5.0 / 3.0 * v[6, 5] / 2.0 - 1.0 / t[7, 6] - \
              5.0 / 3.0 / t[2, 6] - 1.0 / t[6, 10] - 1.0 / t[6, 5]
    F[6, 7] = 5.0 / 3.0 * v[7, 6] / 2.0 + 1.0 / t[7, 6]
    F[6, 10] = 1.0 / t[6, 10]

    F[7, 3] = 5.0 / 3.0 * v[3, 7] / 2.0 + 5.0 / 3.0 / t[3, 7]
    F[7, 6] = -5.0 / 3.0 * v[7, 6] / 2.0 + 1.0 / t[7, 6]
    F[7, 7] = 5.0 / 3.0 * v[3, 7] / 2.0 - 5.0 / 3.0 * v[7, 6] / 2.0 - \
              5.0 / 3.0 / t[3, 7] - 1.0 / t[7, 11] - 1.0 / t[7, 6]
    F[7, 11] = 1.0 / t[7, 11]

    F[8, 4] = 3.0 / 2.0 / t[4, 8]
    F[8, 8] = -1.0 / t[9, 8] - 3.0 / 2.0 / t[4, 8]
    F[8, 9] = 1.0 / t[9, 8]

    F[9, 5] = 3.0 / 2.0 / t[5, 9]
    F[9, 8] = 1.0 / t[9, 8]
    F[9, 9] = -1.0 / t[10, 9] - 3.0 / 2.0 / t[5, 9] - 1.0 / t[9, 8]
    F[9, 10] = 1.0 / t[10, 9]

    F[10, 6] = 3.0 / 2.0 / t[6, 10]
    F[10, 9] = 1.0 / t[10, 9]
    F[10, 10] = -1.0 / t[11, 10] - 3.0 / 2.0 / t[6, 10] - 1.0 / t[10, 9]
    F[10, 11] = 1.0 / t[11, 10]

    F[11, 7] = 3.0 / 2.0 / t[7, 11]
    F[11, 10] = 1.0 / t[11, 10]
    F[11, 11] = -3.0 / 2.0 / t[7, 11] - 1.0 / t[11, 10]

    return F


@njit()
def model(ic, q, mol_mass, lifetime,
          F, temp, oh, cl,
          arr_oh=np.array([1.0e-30, -1800.0]),
          arr_cl=np.array([1.0e-30, -1800.0]),
          dt=2.*24.*3600.,
          mass=5.1170001e+18 * 1000 * np.array([0.125, 0.125, 0.125, 0.125,
                                                0.075, 0.075, 0.075, 0.075,
                                                0.050, 0.050, 0.050, 0.050]),
          nsteps=-1
          ):
    """Main model code

    Parameters
    ----------
    ic : ndarray
        1d, n_box
        Initial conditions of each boxes (pmol/mol) (~pptv).
    q : ndarray
        2d, n_months x n_box_surface
        Monthly emissions into the surface boxes (Gg/yr).
        The length of the simulation is determined by the length of q.
    mol_mass : float
        Single value of molecular mass (g/mol).
    lifetime : ndarray
        2d, n_months x n_box
        Non-OH first-order lifetimes for each month in each box (years).
    F : ndarray
        3d, month x n_box x n_box
        Transport parameter matrix.
    temp : ndarray
        2d, month x n_box
        Temperature (K).
    oh, cl : ndarray
        2d, month x n_box
        Chlorine and OH radical concentrations (molec cm^-3).
    arr_OH, arr_Cl : ndarray, optional
        1d, 2
        Arrhenius A and E/R constants for (X + sink) reactions.
    mass : ndarray
        1d, n_box
        Air mass of individual boxes.
    dt : float, optional
        Delta time (s).
    nsteps : int
        Number of timesteps to run since simulation start (default=-1,
        which ignores this argument)

    Returns
    -------
    c_month : ndarray
        2d, n_months x output_boxes
        Mole fractions (pmol/mol).
    burden : ndarray
        2d, n_months x n_box
        The monthly-average global burden (g).
    emissions : ndarray
        2d, n_months x n_box
        The monthly-average mass emissions (g).
    losses : ndarray
        3d, n_resolved_losses x n_months x n_box.
        The monthly-average mass loss (g).
    lifetimes : dict
        3d, (n_resolved_losses+total) x n_months x n_box.
        Lifetimes calculated from individual time steps (year).
    
    """

    # =========================================================================
    #     Set constants
    # =========================================================================
    # Constants
    day_to_sec = 24.0 * 3600.0
    year_to_sec = 365.25 * day_to_sec
    mol_m_air = 28.97  # dry molecular mass of air in g/mol

    n_months = len(q)
    n_box = len(mass)

    # =========================================================================
    #     Test array sizes
    # =========================================================================
    if F.shape[0] != n_months:
        raise Exception("Error: number of months in F and q don't match")
    if temp.shape[0] != n_months:
        raise Exception("Error: number of months in temp and q don't match")
    if oh.shape[0] != n_months:
        raise Exception("Error: number of months in oh and q don't match")
    if cl.shape[0] != n_months:
        raise Exception("Error: number of months in cl and q don't match")

    # =========================================================================
    #     Process input data
    # =========================================================================

    # Emissions in g/s
    q_gs = np.zeros((n_months, n_box))
    q_gs[:, 0:len(q[0])] = q * 1.0e9 / year_to_sec  # g/s

    # Sub-time-step
    mi_ti = 30.0 * day_to_sec / dt
    if 30.0 % mi_ti == 0.0:
        mi_ti = int(mi_ti)
    else:
        raise Exception("Value Error: dt is not a fraction of 30 days")

    # Initial burden (c in g)
    c = ic * 1.0e-12 * mol_mass / mol_m_air * mass  # ppt to g

    # Start and end timesteps
    # TODO: Allow non-zero start (and restart file)
    start_ti = 0
    if nsteps > 0:
        end_ti = start_ti + nsteps
        if end_ti > n_months * mi_ti:
            end_ti = n_months * mi_ti
    else:
        end_ti = n_months * mi_ti

    # In 12-box model world, there are 360 days in a year
    dt_scaled = dt * 365. / 360.

    # Loss factors (i.e. independent of c at this stage)
    # Unit-less (will multiply c to get loss in g)
    loss_oh_factor = dt_scaled * arr_oh[0] * np.exp(arr_oh[1] / temp) * oh
    loss_cl_factor = dt_scaled * arr_cl[0] * np.exp(arr_cl[1] / temp) * cl
    loss_other_factor = dt_scaled / lifetime / year_to_sec

    # Time-integrated emissions for each step (g)
    q_g = dt_scaled * q_gs

    # =========================================================================
    #     Output Arrays
    # =========================================================================
    c_month = np.zeros((n_months, n_box))
    cnt_month = np.zeros((n_months, n_box))
    cnt_global_month = np.zeros(n_months)

    loss_oh = np.zeros((n_months, n_box))
    loss_cl = np.zeros((n_months, n_box))
    loss_other = np.zeros((n_months, n_box))

    lifetime_oh = np.zeros((n_months, n_box))
    lifetime_cl = np.zeros((n_months, n_box))
    lifetime_other = np.zeros((n_months, n_box))
    lifetime_total = np.zeros((n_months, n_box))

    global_lifetime_oh = np.zeros(n_months)
    global_lifetime_cl = np.zeros(n_months)
    global_lifetime_strat = np.zeros(n_months)
    global_lifetime_othertrop = np.zeros(n_months)
    global_lifetime_othertroplower = np.zeros(n_months)
    global_lifetime_total = np.zeros(n_months)

    """
    Run model
    ti is instantaneous timestep
    mi is month index (0 - 11)
    """
    for ti in range(start_ti, end_ti):

        # Determine if we're at the start of the month
        mi_ti_cnt = ti % mi_ti

        if not mi_ti_cnt:
            # Month index
            mi = int(ti / mi_ti)

            # Month specific constants
            loss_oh_factor_mi = loss_oh_factor[mi]
            loss_cl_factor_mi = loss_cl_factor[mi]
            loss_other_factor_mi = loss_other_factor[mi]
            q_mi = q_g[mi]
            F_mi = F[mi]

            # Initialise running totals
            c_mi = np.zeros(n_box)
            cnt_mi = np.zeros(n_box)

            loss_oh_mi = np.zeros(n_box)
            loss_cl_mi = np.zeros(n_box)
            loss_other_mi = np.zeros(n_box)

            lifetime_oh_mi = np.zeros(n_box)
            lifetime_cl_mi = np.zeros(n_box)
            lifetime_other_mi = np.zeros(n_box)
            lifetime_total_mi = np.zeros(n_box)

            global_lifetime_oh_mi = 0.
            global_lifetime_cl_mi = 0.
            global_lifetime_strat_mi = 0.
            global_lifetime_othertrop_mi = 0.
            global_lifetime_othertroplower_mi = 0.
            global_lifetime_total_mi = 0.

        # Step forward solver
        c = model_solver_rk4(c / mass, F_mi, dt) * mass

        # Loss (g)
        loss_oh_ti = c * loss_oh_factor_mi
        loss_cl_ti = c * loss_cl_factor_mi
        loss_other_ti = c * loss_other_factor_mi

        loss_oh_ti[loss_oh_ti == 0.0] = 1.0e-24
        loss_cl_ti[loss_cl_ti == 0.0] = 1.0e-24
        loss_other_ti[loss_other_ti == 0.0] = 1.0e-24

        # Update burden
        c += q_mi - loss_oh_ti - loss_cl_ti - loss_other_ti

        # Add to monthly totals (number of timesteps will be divided out later, for quantities that need averaging)
        ######################################
        c_mi += c

        loss_oh_mi += loss_oh_ti  # g
        loss_cl_mi += loss_cl_ti  # g
        loss_other_mi += loss_other_ti  # g

        # lifetimes in s
        lifetime_oh_mi += c / loss_oh_ti * dt_scaled
        lifetime_cl_mi += c / loss_cl_ti * dt_scaled
        lifetime_other_mi += c / loss_other_ti * dt_scaled
        lifetime_total_mi += c / (loss_oh_ti + loss_cl_ti + loss_other_ti) * dt_scaled

        global_lifetime_oh_mi += c.sum() / max([loss_oh_ti.sum(), 1.0e-30]) * dt_scaled
        global_lifetime_cl_mi += c.sum() / max([loss_cl_ti.sum(), 1.0e-30]) * dt_scaled
        global_lifetime_strat_mi += c.sum() / max([loss_other_ti[8:].sum(), 1.0e-30]) * dt_scaled
        global_lifetime_othertrop_mi += c.sum() / max([loss_other_ti[0:8].sum(), 1.0e-30]) * dt_scaled
        global_lifetime_othertroplower_mi += c.sum() / max([loss_other_ti[0:4].sum(), 1.0e-30]) * dt_scaled
        global_lifetime_total_mi += c.sum() / max([(loss_oh_ti + loss_cl_ti + loss_other_ti).sum(), 1.0e-30]) * dt_scaled

        cnt_mi += 1

        if mi_ti_cnt == mi_ti - 1:
            # Write Monthly totals for averages
            c_month[mi] = c_mi
            cnt_month[mi] += cnt_mi
            cnt_global_month[mi] += cnt_mi[0]

            loss_oh[mi] = loss_oh_mi
            loss_cl[mi] = loss_cl_mi
            loss_other[mi] = loss_other_mi

            lifetime_oh[mi] = lifetime_oh_mi
            lifetime_cl[mi] = lifetime_cl_mi
            lifetime_other[mi] = lifetime_other_mi
            lifetime_total[mi] = lifetime_total_mi

            global_lifetime_oh[mi] = global_lifetime_oh_mi
            global_lifetime_cl[mi] = global_lifetime_cl_mi
            global_lifetime_strat[mi] = global_lifetime_strat_mi
            global_lifetime_othertrop[mi] = global_lifetime_othertrop_mi
            global_lifetime_othertroplower[mi] = global_lifetime_othertroplower_mi
            global_lifetime_total[mi] = global_lifetime_total_mi

    # Calculate monthly averages
    burden_out = c_month.copy()
    burden_out = burden_out / cnt_month

    mole_fraction_out = np.divide(burden_out / mol_mass * mol_m_air * 1.0e12, mass)  # ppt

    # Emissions in g
    q_out = q_g * mi_ti  # g

    # Losses in g
    losses = {"oh": loss_oh, "cl": loss_cl, "other": loss_other}

    # Global lifetimes in years
    global_lifetimes = {"global_oh": global_lifetime_oh / cnt_global_month / year_to_sec,
                        "global_cl": global_lifetime_cl / cnt_global_month / year_to_sec,
                        "global_strat": global_lifetime_strat / cnt_global_month / year_to_sec,
                        "global_othertrop": global_lifetime_othertrop / cnt_global_month / year_to_sec,
                        "global_othertroplower": global_lifetime_othertroplower / cnt_global_month / year_to_sec,
                        "global_total": global_lifetime_total / cnt_global_month / year_to_sec}
    
    # Local lifetimes in years
    lifetimes = {"oh": lifetime_oh / cnt_month / year_to_sec,
                 "cl": lifetime_cl / cnt_month / year_to_sec,
                 "other": lifetime_other / cnt_month / year_to_sec}

    return mole_fraction_out, burden_out, q_out, losses, global_lifetimes
