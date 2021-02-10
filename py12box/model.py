"""
Copyright 2021 Matt Rigby

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from py12box import startup
import numpy as np
from pathlib import Path


class Model:
    #TODO: Add docstring

    # Mass of the atmosphere
    mass=5.1170001e+18 * 1000 * np.array([0.125, 0.125, 0.125, 0.125,
                                          0.075, 0.075, 0.075, 0.075,
                                          0.050, 0.050, 0.050, 0.050])

    def __init__(self, species, project_directory,
                 species_param_file=None):

        mol_mass, oh_a, oh_er, units = startup.get_species_parameters(species,
                                                                      param_file=species_param_file)
        self.mol_mass = mol_mass
        self.oh_a = oh_a
        self.oh_er = oh_er
        self.units = units

        time, emissions, ic, lifetime = startup.get_case_parameters(species, project_directory)
        self.time = time
        self.emissions = emissions
        self.ic = ic
        self.lifetime = lifetime

        _i_t, _i_v1, _t, _v1, oh, cl, temperature = startup.get_model_parameters(len(emissions/12),
                                                                                 input_dir=Path("data/inputs"))
        self.oh = oh
        self.cl = cl
        self.temperature = temperature

        self.F = startup.transport_matrix(_i_t, _i_v1, _t, _v1)


if __name__ == "__main__":

    mod = Model("CFC-11", Path("data/example"))

    mod.mol_mass


