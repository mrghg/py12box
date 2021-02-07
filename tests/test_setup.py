import py12box.setup as setup
import numpy as np

def test_get_species_parameters():
    '''
    Test setup.get_species_parameters

    Ensure that sensible species info is returned
    '''

    mol_mass, OH_A, OH_ER = setup.get_species_parameters("CFC-11")

    assert np.allclose(mol_mass, 137.3688, rtol=0.001)
