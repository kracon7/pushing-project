'''
Test hidden state simulator
'''

import os
import sys
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

SIM_STEPS = 50

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    hidden_sim = HiddenStateSimulator(param_file)
    sim = GraspNRotateSimulator(param_file)

    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)
    mapping = np.zeros(sim.ngeom).astype("int")
    hidden_state = hidden_sim.map_to_hidden_state(mass_gt, mapping, friction_gt, mapping)

    # Control input
    u = {4: [[10, 0, 1.5] for _ in range(60)],
         40: [[0, -10, -1.5] for _ in range(60)],
         15: [[-10, 0, 0.5] for _ in range(60)]}

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEPS-1, 1)]

    sim.input_parameters(mass_gt, mapping, friction_gt, mapping, u, loss_steps)
    sim.run(SIM_STEPS, render=True)

    hidden_sim.input_parameters(hidden_state["body_mass"], hidden_state["body_inertia"], 
                hidden_state["body_com"], hidden_state["si"], mapping, u, loss_steps)
    hidden_sim.run(SIM_STEPS, render=True)

    geom_pos_1 = sim.geom_pos.to_numpy()
    geom_pos_2 = hidden_sim.geom_pos.to_numpy()

    error = np.linalg.norm(geom_pos_1 - geom_pos_2)
    print("Normal of the difference in geom pos: %.8f"%error)