import numpy as np
from pathlib import Path

from py12box.model import Model
from  py12box import startup


def test_get_species_parameters():
    '''
    Test setup.get_species_parameters

    Ensure that sensible species info is returned
    '''

    mol_mass, oh_a, oh_er, unit = startup.get_species_parameters("CFC-11")

    assert np.isclose(mol_mass, 137.3688, rtol=0.001)
    assert np.isclose(oh_a, 1e-12, rtol=0.001)
    assert np.isclose(oh_er, -3700, rtol=0.001)
    assert unit == 1e-12


def test_get_case_parameters():
    time, emissions, ic, lifetime = startup.get_case_parameters("CFC-11",
                                                                Path("data/example"))
    assert time[0] == 1990.


def test_model():

    box_mod = Model("HFC-134a", Path("data/example"))

    assert np.isclose(box_mod.mol_mass, 102.0311, rtol=0.001)
    assert np.isclose(box_mod.oh_a, 1.03E-12, rtol=0.001)
    assert np.isclose(box_mod.oh_er, -1620, rtol=0.001)
    assert box_mod.units == 1e-12
    
