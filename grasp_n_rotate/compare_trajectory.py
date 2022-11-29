'''
Plot the body pose trajectories of objects with close mass and friction

Assumptions:
    -- uniform friction profile   1-D mu
    -- uniform mass   1-D m
'''

import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import taichi as ti
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

SIM_STEPS = 400

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)

    mapping = np.zeros(sim.ngeom)
    u = [[4, 5, 0, 1] for _ in range(100)] + \
        [[40, 0, 3, 0.5] for _ in range(200)] + \
        [[10, 1.7, 0, 1] for _ in range(200)]

    fig = make_subplots(
        rows=3, cols=1, subplot_titles=('Px', 'Py', 'Pw'),
        horizontal_spacing=0.02, vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.33]
    )
    fig.update_xaxes(title="time (s)", row=3, col=1)
    fig.update_yaxes(title="Px (m)", row=1, col=1)
    fig.update_yaxes(title="Py (m)", row=2, col=1)
    fig.update_yaxes(title="Pw (rad)", row=3, col=1)

    states = [[0.2, 0.115],
              [0.3, 0.11],
              [0.5, 0.1],
              [0.75, 0.09],
              [0.8, 0.0875]]
    
    for i, state in enumerate(states):
        mass = state[1] * np.ones(sim.block_object.num_particle)
        friction = state[0] * np.ones(sim.block_object.num_particle)
        sim.input_parameters(mass, mapping, friction, mapping, u)
        sim.run(SIM_STEPS)
        qpos, rpos = sim.body_qpos.to_numpy()[:SIM_STEPS], sim.body_rpos.to_numpy()[:SIM_STEPS]
        prefix = '%.2f_%.3f_'%(state[0], state[1])
        fig.add_trace(go.Scatter(y=qpos[:, 0], name=prefix+'px', legendgroup='1'), row=1, col=1)
        fig.add_trace(go.Scatter(y=qpos[:, 1], name=prefix+'py', legendgroup='2'), row=2, col=1)
        fig.add_trace(go.Scatter(y=rpos, name=prefix+'pw', legendgroup='3'), row=3, col=1)


    fig.update_layout(
        legend_tracegroupgap = 180
    )
    fig.show()