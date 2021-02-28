import numpy as np
from pathlib import Path

from py12box.model import Model
from  py12box import startup, get_data


def test_get_species_parameters():
    mol_mass, oh_a, oh_er, unit = startup.get_species_parameters("CFC-11")

    assert np.isclose(mol_mass, 137.3688, rtol=0.001)
    assert np.isclose(oh_a, 1e-12, rtol=0.001)
    assert np.isclose(oh_er, -3700, rtol=0.001)
    assert unit == 1e-12


def test_get_emissions():
    time, emissions = startup.get_emissions("CFC-11",
                                            get_data("example/CFC-11"))
    
    assert time[0] == 1990.
    assert len(time) == 29*12
    assert emissions[0, 0] == 100.
    assert emissions[-1, 3] == 0
    assert emissions.shape == (348, 4)

def test_get_lifetime():
    lifetime = startup.get_lifetime("CFC-11",
                                    get_data("example/CFC-11"), n_years=4)
    assert lifetime[0, 8] == 36.878456
    assert lifetime.shape == (12*4, 12)

def test_get_lifetime_nofile():
    # for now, just get the path to the data directory,
    # which DOESN'T have a lifetime file in
    lifetime = startup.get_lifetime("CFC-11",
                                    get_data("example/CFC-11_nolifetime"), 
                                    n_years=4)


def test_get_initial_conditions():
    ic = startup.get_initial_conditions("CFC-11",
                                        get_data("example/CFC-11"))
    assert ic[0] == 200.
    assert len(ic) == 12


def test_get_model_parameters():

    i_t, i_v1, t, v1, oh, cl, temperature = startup.get_model_parameters(2,
                                                                         input_dir=get_data("inputs"))

    assert len(i_t) == 17
    assert i_t[10][0] == 8
    assert i_t[10][1] == 4
    assert len(i_v1) == 10
    assert i_v1[5][0] == 6
    assert i_v1[5][1] == 7
    assert t.shape == (24, 17)
    assert t[6, 7] == 38.
    assert v1.shape == (24, 10)
    assert v1[7, 8] == -54.
    assert oh.shape == (24, 12)
    assert np.isclose(oh[2, 2], 1810513., rtol = 0.001)
    assert cl.shape == (24, 12)
    assert temperature.shape == (24, 12)
    assert np.isclose(temperature.max(), 283.78165)
    assert np.isclose(temperature.min(), 207.19719)

    F = startup.transport_matrix(i_t, i_v1, t, v1)

    assert F.shape == (24, 12, 12)
    assert F[0, 5, 6] == F[12, 5, 6]
    assert np.isclose(F[2, 2, 2], -3.98587280261696e-07, rtol = 0.0001)


def test_model_class():

    box_mod = Model("HFC-134a", get_data("example/HFC-134a"))

    assert np.isclose(box_mod.mol_mass, 102.0311, rtol=0.001)
    assert np.isclose(box_mod.oh_a, 1.03E-12, rtol=0.001)
    assert np.isclose(box_mod.oh_er, -1620, rtol=0.001)
    assert box_mod.units == 1e-12


# TODO: Add tests for core model
if __name__ == "__main__":
    test_get_lifetime()