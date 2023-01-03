'''
Hidden state si loss contour and gradient direction plot with hidden state simulator
'''

import os
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults

SIM_STEP = 50

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = HiddenStateSimulator(param_file)

    mass = 0.1 * np.ones(sim.block_object.num_particle)
    friction = 0.5 * np.ones(sim.block_object.num_particle)
    n_partition = 6
    mass_mapping = np.repeat(np.arange(n_partition), sim.ngeom//n_partition).astype("int")
    friction_mapping = np.zeros(sim.block_object.num_particle).astype("int")
    
    hidden_state_mapping = HiddenStateMapping(sim)

    hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                    friction, friction_mapping)
    print(hidden_state)
    explicit_state = hidden_state_mapping.map_to_explicit_state(hidden_state, 
                                            mass_mapping, friction_mapping)
    
    print(explicit_state)
