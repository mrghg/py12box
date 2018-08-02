# -*- coding: utf-8 -*-
"""
core.py
AGAGE Box model

The model calculates semi-hemispheric monthly average mole fractions, based on
given emissions, stratospheric and oceanic lifetimes and sink reaction rates.

This code started as a python 3 port of the IDL code written by Matt Rigby
available from: https://bitbucket.org/mrghg/agage-12-box-model/

Major differences include:
    01. Aside from model_transport_matrix function, the model is flexible to
        have any number of boxes, not just 12(3x4).
    02. All scaling must be preprocessed. The model does not take scaling
        factors
    03. Lifetime is monthly not seasonal.

Usage:
    01. Can be imported to a python script or ipython session
            import agage-box-model.core
        and run the functions individually.

Notes:
    01. The text area between import and function definitions are configurable.
    02. The input .csv files must include linefeeds (\n) as the line-break.
        Python reader ignores carriage returns (\r) used by old Macs.
        Windows style (\r\n) is fine.

Initial author: Edward Chung (s1765003@sms.ed.ac.uk)
Version History
1.0 20171026    EC  Initial code.
2.0 20171222    EC  Optimisation and addition of chlorine loss term.
3.0 20180106    EC  Inclusion of numba.jit.
3.1 20180116    EC  Function name changes; docstring update;
                    and use of numpy save files.
4.0 20180406    EC  Removed ability to run on its own, to become more Pythonic.
"""
from numba import jit, njit
import numpy as np


@njit()
def model_solver_RK4(chi, F, dt):
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
    B = np.dot(F, (chi + dt*A/2.0))
    C = np.dot(F, (chi + dt*B/2.0))
    D = np.dot(F, (chi + dt*C))
    chi += dt/6.0*(A + 2.0*(B + C) + D)
    return chi


@jit(nopython = True)
def model_transport_matrix(i_t, i_v1, t_in, v1_in):
    '''Calculate transport matrix
    Based on equations in:
        Cunnold, D. M. et al. (1983).
        The Atmospheric Lifetime Experiment 3. Lifetime Methodology and
        Application to Three Years of CFCl3 Data.
        Journal of Geophysical Research, 88(C13), 8379-8400.

    This function outputs a 12x12 matrix (F), calculated by collecting terms in
    the full equation scheme written out on doc_F_equation.txt.
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

    '''
    F = np.zeros((12, 12))
    t = np.zeros((12, 12))
    v = np.zeros((12, 12))

    for i in range(0, len(i_v1)):
        v[i_v1[i, 1], i_v1[i, 0]] = 1.0/v1_in[i]
    for i in range(0, len(i_t)):
        t[i_t[i, 1], i_t[i, 0]] = t_in[i]

    F[0, 0] = v[1, 0]/2.0 - v[0, 4]/2.0 - 1.0/t[1, 0] - 1.0/t[0, 4]
    F[0, 1] = v[1, 0]/2.0 + 1.0/t[1, 0]
    F[0, 4] = - v[0, 4]/2.0 + 1.0/t[0, 4]

    F[1, 0] = - v[1, 0]/2.0 + 1.0/t[1, 0]
    F[1, 1] = v[2, 1]/2.0 - v[1, 5]/2.0 - v[1, 0]/2.0 - \
                         1.0/t[2, 1] - 1.0/t[1, 5] - 1.0/t[1, 0]
    F[1, 2] = v[2, 1]/2.0 + 1.0/t[2, 1]
    F[1, 5] = - v[1, 5]/2.0 + 1.0/t[1, 5]

    F[2, 1] = -v[2, 1]/2.0 + 1.0/t[2, 1]
    F[2, 2] = v[3, 2]/2.0 - v[2, 6]/2.0 - v[2, 1]/2.0 - \
              1.0/t[3, 2] - 1.0/t[2, 6] - 1.0/t[2, 1]
    F[2, 3] = v[3, 2]/2.0 + 1.0/t[3, 2]
    F[2, 6] = -1.0*v[2, 6]/2.0 + 1.0/t[2, 6]

    F[3, 2] = -v[3, 2]/2.0 + 1.0/t[3, 2]
    F[3, 3] = -v[3, 7]/2.0 - v[3, 2]/2.0 - 1.0/t[3, 7] - 1.0/t[3, 2]
    F[3, 7] = -v[3, 7]/2.0 + 1.0/t[3, 7]

    F[4, 0] = 5.0/3.0*v[0, 4]/2.0 + 5.0/3.0/t[0, 4]
    F[4, 4] = 5.0/3.0*v[5, 4]/2.0 + 5.0/3.0*v[0, 4]/2.0 - \
              1.0/t[5, 4] - 5.0/3.0/t[0, 4] - 1.0/t[4, 8]
    F[4, 5] = 5.0/3.0*v[5, 4]/2.0 + 1.0/t[5, 4]
    F[4, 8] = 1.0/t[4, 8]

    F[5, 1] = 5.0/3.0*v[1, 5]/2.0 + 5.0/3.0/t[1, 5]
    F[5, 4] = -5.0/3.0*v[5, 4]/2.0 + 1.0/t[5, 4]
    F[5, 5] = 5.0/3.0*v[6, 5]/2.0 + 5.0/3.0*v[1, 5]/2.0 - \
              5.0/3.0*v[5, 4]/2.0 - 1.0/t[6, 5] - \
              5.0/3.0/t[1, 5] - 1.0/t[5, 9] - 1.0/t[5, 4]
    F[5, 6] = 5.0/3.0*v[6, 5]/2.0 + 1.0/t[6, 5]
    F[5, 9] = 1.0/t[5, 9]

    F[6, 2] = 5.0/3.0*v[2, 6]/2.0 + 5.0/3.0/t[2, 6]
    F[6, 5] = -5.0/3.0*v[6, 5]/2.0 + 1.0/t[6, 5]
    F[6, 6] = 5.0/3.0*v[7, 6]/2.0 + 5.0/3.0*v[2, 6]/2.0 - \
              5.0/3.0*v[6, 5]/2.0 - 1.0/t[7, 6] - \
              5.0/3.0/t[2, 6] - 1.0/t[6, 10] - 1.0/t[6, 5]
    F[6, 7] = 5.0/3.0*v[7, 6]/2.0 + 1.0/t[7, 6]
    F[6, 10] = 1.0/t[6, 10]

    F[7, 3] = 5.0/3.0*v[3, 7]/2.0 + 5.0/3.0/t[3, 7]
    F[7, 6] = -5.0/3.0*v[7, 6]/2.0 + 1.0/t[7, 6]
    F[7, 7] = 5.0/3.0*v[3, 7]/2.0 - 5.0/3.0*v[7, 6]/2.0 - \
              5.0/3.0/t[3, 7] - 1.0/t[7, 11] - 1.0/t[7, 6]
    F[7, 11] = 1.0/t[7, 11]

    F[8, 4] = 3.0/2.0/t[4, 8]
    F[8, 8] = -1.0/t[9, 8] - 3.0/2.0/t[4, 8]
    F[8, 9] = 1.0/t[9, 8]

    F[9, 5] = 3.0/2.0/t[5, 9]
    F[9, 8] = 1.0/t[9, 8]
    F[9, 9] = -1.0/t[10, 9] - 3.0/2.0/t[5, 9] - 1.0/t[9, 8]
    F[9, 10] = 1.0/t[10, 9]

    F[10, 6] = 3.0/2.0/t[6, 10]
    F[10, 9] = 1.0/t[10, 9]
    F[10, 10] = -1.0/t[11, 10] - 3.0/2.0/t[6, 10] - 1.0/t[10, 9]
    F[10, 11] = 1.0/t[11, 10]

    F[11, 7] = 3.0/2.0/t[7, 11]
    F[11, 10] = 1.0/t[11, 10]
    F[11, 11] = -3.0/2.0/t[7, 11] - 1.0/t[11, 10]
    
    return F



@njit()
def model(ic, q, mol_mass, lifetime,
              F, temp, OH, Cl,
              arr_OH = np.array([1.0e-30, 1800.0]),
              arr_Cl = np.array([1.0e-30, 1800.0]),
              dt = 2.0*24.0*3600.0,
              mass = 5.1170001e+18*1000*np.array([0.125, 0.125, 0.125, 0.125,
                                                  0.075, 0.075, 0.075, 0.075,
                                                  0.050, 0.050, 0.050, 0.050])
              ):
    '''Main model code
    ic : ndarray
        1d, n_box
        Initial conditions of each boxes (pmol/mol) (~pptv).
    q : ndarray
        2d, n_months x n_box_surface
        Monthly emissions into the surface boxes (Gg/yr).
        The length of the simulation is determined by the length of q.
    mol_mass : float
        Single value of molecular mass (g/mol).
    arr_OH, arr_Cl : ndarray
        1d, 2
        Arrhenius A and E/R constants for (X + sink) reactions.
    mass : ndarray
        1d, n_box
        Air mass of individual boxes.
    lifetime : ndarray
        2d, n_months x n_box
        Non-OH first-order lifetimes for each month in each box (years).
    dt : float
        Delta time (s).
    F : ndarray
        3d, month x n_box x n_box
        Transport parameter matrix.
    OH, Cl : ndarray
        2d, month x n_box
        Chlorine and OH radical concentrations (molec cm^-3).
    temp : ndarray
        2d, month x n_box
        Temperature (K).

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
    '''
    # =========================================================================
    #     Set constants
    # =========================================================================
    # Constants
    day_to_sec = 24.0*3600.0
    year_to_sec = 365.25*day_to_sec
    mol_m_air = 28.97  # dry molecular mass of air in g/mol

    n_months = len(q)
    n_box = len(mass)

    # =========================================================================
    #     Process input data
    # =========================================================================
    # Arrhenius constants: assume inert if no specified
    #arr_A_OH, arr_ER_OH = get_arr(arr_OH, np.array([1.0e-30, 1800.0]))
    #arr_A_Cl, arr_ER_Cl = get_arr(arr_Cl, np.array([1.0e-30, 1800.0]))

    # Emissions
    q_model = np.zeros((n_months, n_box))
    q_model[:, 0:len(q[0])] = q*1.0e9/year_to_sec  # g/s

    # Sub-time-step
    mi_ti = 30.0*day_to_sec/dt
    if 30.0 % mi_ti == 0.0:
        mi_ti = int(mi_ti)
    else:
        raise Exception("Value Error: dt is not a fraction of 30 days")

    # Initial conditions and time settings
    c = ic*1.0e-12*mol_mass/mol_m_air*mass  # ppt to g

    start_ti = 0
    end_ti = n_months*mi_ti

    dt_scaled = dt*365.0/360.0

    # Multiply factors
    loss_OH_factor = dt_scaled*arr_OH[0]*np.exp(-arr_OH[1]/temp)*OH
    loss_Cl_factor = dt_scaled*arr_Cl[0]*np.exp(-arr_OH[1]/temp)*Cl
    loss_other_factor = dt_scaled/lifetime/year_to_sec
    emissions_dt = dt_scaled*q_model


    # =========================================================================
    #     Output Array
    # =========================================================================
    c_month = np.zeros((n_months, n_box))
    cnt_month = np.zeros((n_months, n_box))

    burden = np.zeros((n_months, n_box))

    emissions = np.zeros((n_months, n_box))

    loss_OH = np.zeros((n_months, n_box))
    loss_Cl = np.zeros((n_months, n_box))
    loss_other = np.zeros((n_months, n_box))

    lifetime_OH = np.zeros((n_months, n_box))
    lifetime_Cl = np.zeros((n_months, n_box))
    lifetime_other = np.zeros((n_months, n_box))
    lifetime_total = np.zeros((n_months, n_box))

    # =========================================================================
    #     Run model
    #       Time stepping: Speed matters, not readability
    #       Hence minimise accessing arrays
    # =========================================================================
    for ti in range(start_ti, end_ti):
        mi_ti_cnt = ti % mi_ti
        if not mi_ti_cnt:
            mi = int(ti/mi_ti)

            # Month specific constants
            loss_OH_factor_mi = loss_OH_factor[mi]
            loss_Cl_factor_mi = loss_Cl_factor[mi]
            loss_other_factor_mi = loss_other_factor[mi]
            emissions_mi = emissions_dt[mi]
            F_mi = F[mi]

            # Initialise totals
            c_mi = np.zeros(n_box)
            cnt_mi = np.zeros(n_box)
            loss_OH_mi = np.zeros(n_box)
            loss_Cl_mi = np.zeros(n_box)
            loss_other_mi = np.zeros(n_box)
            lifetime_OH_mi = np.zeros(n_box)
            lifetime_Cl_mi = np.zeros(n_box)
            lifetime_other_mi = np.zeros(n_box)
            lifetime_total_mi = np.zeros(n_box)

        # Step forward solver
        c = model_solver_RK4(c/mass, F_mi, dt) * mass
        
        # Loss
        loss_OH_ti = c*loss_OH_factor_mi
        loss_Cl_ti = c*loss_Cl_factor_mi
        loss_other_ti = c*loss_other_factor_mi

        # burden
        c += emissions_mi - loss_OH_ti - loss_Cl_ti - loss_other_ti

        # Monthly totals
        c_mi += c

        loss_OH_mi += loss_OH_ti
        loss_Cl_mi += loss_Cl_ti
        loss_other_mi += loss_other_ti

        loss_OH_ti[loss_OH_ti == 0.0] = 1.0e-24
        loss_Cl_ti[loss_Cl_ti == 0.0] = 1.0e-24
        loss_other_ti[loss_other_ti == 0.0] = 1.0e-24

        lifetime_OH_mi += c/loss_OH_ti
        lifetime_Cl_mi += c/loss_Cl_ti
        lifetime_other_mi += c/loss_other_ti
        lifetime_total_mi += c/(loss_OH_ti + loss_Cl_ti + loss_other_ti)

        cnt_mi += 1

        if mi_ti_cnt == mi_ti - 1:
            # Write Monthly totals for averages
            c_month[mi] = c_mi
            cnt_month[mi] += cnt_mi

            loss_OH[mi] = loss_OH_mi
            loss_Cl[mi] = loss_Cl_mi
            loss_other[mi] = loss_other_mi

            lifetime_OH[mi] = lifetime_OH_mi
            lifetime_Cl[mi] = lifetime_Cl_mi
            lifetime_other[mi] = lifetime_other_mi
            lifetime_total[mi] = lifetime_total_mi

    # Calculate monthly averages
    burden = c_month.copy()
    burden = burden/cnt_month

    c_month = burden/mol_mass*mol_m_air*1.0e12  # ppt
    c_month = np.divide(c_month, mass)

    emissions = q_model*30*day_to_sec

    loss_OH = loss_OH/cnt_month
    loss_Cl = loss_Cl/cnt_month
    loss_other = loss_other/cnt_month
    losses = np.zeros((3, n_months, n_box))
    losses[0] = loss_OH
    losses[1] = loss_Cl
    losses[2] = loss_other

    lifetime_OH = lifetime_OH/cnt_month*dt/year_to_sec
    lifetime_Cl = lifetime_Cl/cnt_month*dt/year_to_sec
    lifetime_other = lifetime_other/cnt_month*dt/year_to_sec
    lifetimes = np.zeros((4, n_months, n_box))
    lifetimes[0] = lifetime_OH
    lifetimes[1] = lifetime_Cl
    lifetimes[2] = lifetime_other
    lifetimes[3] = lifetime_total

    return c_month, burden, emissions, loss_other, lifetimes
