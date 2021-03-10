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
