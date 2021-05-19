import numpy as np
from pathlib import Path

from py12box.model import Model
from  py12box import startup, get_data

box_mod = Model("HFC-134a", get_data("example/HFC-134a"))


def test_species_parameters():

    assert np.isclose(box_mod.mol_mass, 102.0311, rtol=0.001)
    assert np.isclose(box_mod.oh_a, 1.03E-12, rtol=0.001)
    assert np.isclose(box_mod.oh_er, -1620, rtol=0.001)
    assert box_mod.units == 1e-12


def test_lifetime():

    assert np.isclose(box_mod.steady_state_lifetime, 13.5, rtol=0.001)
    assert np.isclose(box_mod.steady_state_lifetime_oh, 14.23, rtol=0.001)
    assert np.isclose(box_mod.steady_state_lifetime_strat, 267.0706, rtol=0.001)


def test_emissions():

    assert box_mod.emissions.shape == (348, 4)
    assert len(box_mod.time) == 348
    assert box_mod.emissions[0, 0] == 100.


def test_change_start_year():

    box_mod.change_start_year(2000.)
    assert box_mod.time.shape[0] == int(np.round((box_mod.time[-1] - 2000.)*12) + 1)
    assert np.round(box_mod.time[0]) == 2000.
    assert box_mod.time.shape[0] == box_mod.emissions.shape[0]
    assert box_mod.time.shape[0] == box_mod.lifetime.shape[0]
    assert box_mod.time.shape[0] == box_mod.F.shape[0]
    assert box_mod.time.shape[0] == box_mod.oh.shape[0]
    assert box_mod.time.shape[0] == box_mod.cl.shape[0]
    assert box_mod.time.shape[0] == box_mod.temperature.shape[0]
    if hasattr(box_mod, "mf"):
        assert box_mod.time.shape[0] == box_mod.mf.shape[0]
        assert box_mod.time.shape[0] == box_mod.mf_restart.shape[0]
        assert box_mod.time.shape[0] == box_mod.burden.shape[0]
        for key, val in box_mod.losses.items():
            assert box_mod.time.shape[0] == val.shape[0]
        for key, val in box_mod.instantaneous_lifetimes.items():
            assert box_mod.time.shape[0] == val.shape[0]


def test_change_end_year():

    initial_len = box_mod.time.shape[0]
    initial_end = box_mod.time[-1]

    box_mod.change_end_year(2010.)
    assert box_mod.time.shape[0] == initial_len - int(np.round((initial_end - 2010.)*12)) - 1
    assert np.isclose(box_mod.time[-1], 2010. - 1./12.)
    assert box_mod.time.shape[0] == box_mod.emissions.shape[0]
    assert box_mod.time.shape[0] == box_mod.emissions.shape[0]
    assert box_mod.time.shape[0] == box_mod.F.shape[0]
    assert box_mod.time.shape[0] == box_mod.oh.shape[0]
    assert box_mod.time.shape[0] == box_mod.cl.shape[0]
    assert box_mod.time.shape[0] == box_mod.temperature.shape[0]
    if hasattr(box_mod, "mf"):
        assert box_mod.time.shape[0] == box_mod.mf.shape[0]
        assert box_mod.time.shape[0] == box_mod.mf_restart.shape[0]
        assert box_mod.time.shape[0] == box_mod.burden.shape[0]
        for key, val in box_mod.losses.items():
            assert box_mod.time.shape[0] == val.shape[0]
        for key, val in box_mod.instantaneous_lifetimes.items():
            assert box_mod.time.shape[0] == val.shape[0]

def test_restart():

    assert box_mod.mf_restart.shape == box_mod.mf.shape

    # The restart values should be between the bracketing monthly means
    assert (box_mod.mf[0,:4].mean() <= box_mod.mf_restart[0,:4].mean() <= box_mod.mf[1,:4].mean()) or \
            (box_mod.mf[0,:4].mean() >= box_mod.mf_restart[0,:4].mean() >= box_mod.mf[1,:4].mean())
    assert (box_mod.mf[20,:4].mean() <= box_mod.mf_restart[20,:4].mean() <= box_mod.mf[21,:4].mean()) or \
            (box_mod.mf[20,:4].mean() >= box_mod.mf_restart[20,:4].mean() >= box_mod.mf[21,:4].mean())


#TODO: Write test for SF6