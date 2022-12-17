'''
Plot the loss contour over the state space

Assumptions:
    -- uniform friction profile
    -- non-uniform mass
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
    sim = GraspNRotateSimulator(param_file)
    sim_gt = GraspNRotateSimulator(param_file)

    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)

    # map mass and friction to regions 
    mapping = np.zeros(sim.ngeom)

    # Control input
    u = {4: [[10, 0, 1.5] for _ in range(60)],
         40: [[0, -10, -1.5] for _ in range(60)],
         15: [[-10, 0, 0.5] for _ in range(60)]}

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEPS-1, 1)]

    sim_gt.input_parameters(mass_gt, mapping, friction_gt, mapping, u, loss_steps)
    
    # Ground truth forward simulation
    sim_gt.run(SIM_STEPS, render=True)
    geom_pos_gt = sim_gt.geom_pos.to_numpy()

    state_space = np.meshgrid(np.linspace(0.08, 0.12, 33), np.linspace(0.2, 0.8, 25), indexing='ij')
    n1, n2 = state_space[0].shape
    loss_grid = np.zeros((n1, n2))
    grad_grid = np.zeros((n1, n2, 2))
    # Run sim on different mass state and compute loss
    for i in range(n1):
        for j in range(n2):
            mass = state_space[0][i, j] * np.ones(sim.block_object.num_particle)
            friction = state_space[1][i, j] * np.ones(sim.block_object.num_particle)
            sim.input_parameters(mass, mapping, friction, mapping, u, loss_steps)

            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEPS)
                sim.compute_loss(geom_pos_gt)

            print('mass: %4f, friction %4f, loss: %.9f, dl/dx: %.5f, %.5f'%(mass[0], friction[0],
                    sim.loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_friction.grad.to_numpy()[0]))

            grad = [sim.composite_mass.grad.to_numpy()[0], 
                    sim.composite_friction.grad.to_numpy()[0]]

            loss_grid[i, j] = sim.loss[None]
            grad_grid[i, j] = - np.array(grad) / np.linalg.norm(grad)

    # grad_grid = grad_grid.reshape(-1, 2)

    f = ff.create_quiver(state_space[1], state_space[0], grad_grid[:,:,1], grad_grid[:,:,0],
                        scale=0.001)
    trace1 = f.data[0]
    trace2 = go.Contour(z=loss_grid, y0=0.08, dy=0.00125, x0=0.2, dx=0.025,
                            contours=dict(
                                start=0,
                                end=5,
                                size=0.01,
                            )
                        )
    # Plot the loss contour
    fig = go.Figure(data = [trace1, trace2])
    fig.show()

# import plotly.figure_factory as ff
# import plotly.graph_objs as go
# import numpy as np

# x,y = np.meshgrid(np.arange(0, 4, .2), np.arange(0, 4, .2))
# u = np.cos(x)*y
# v = np.sin(x)*y

# f = ff.create_quiver(x, y, u, v)
# trace1 = f.data[0]
# trace2 = go.Contour(
#        z=[[10, 10.625, 12.5, 15.625, 20],
#           [5.625, 6.25, 8.125, 11.25, 15.625],
#           [2.5, 3.125, 5., 8.125, 12.5],
#           [0.625, 1.25, 3.125, 6.25, 10.625],
#           [0, 0.625, 2.5, 5.625, 10]]
#    )
# data=[trace1,trace2]
# fig = go.Figure(data=data)
# fig.show()