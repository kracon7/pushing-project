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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import taichi as ti
from scipy.spatial.transform import Rotation
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
    sim_gt = GraspNRotateSimulator(param_file)
    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)

    # map mass and friction to regions 
    mapping = np.zeros(sim.ngeom)

    # Control input
    u = [[4, 5, 0, 1] for _ in range(100)] + \
        [[40, 0, 3, 0.5] for _ in range(200)] + \
        [[10, 1.7, 0, 0.1] for _ in range(200)]

    sim_gt.input_parameters(mass_gt, mapping, friction_gt, mapping, u)
    
    # Ground truth forward simulation
    sim_gt.run(SIM_STEPS)

    # Time steps to include in loss computation
    loss_steps = [50, 100, 150, 200, 250, SIM_STEPS-1]

    state_space = np.meshgrid(np.linspace(0.08, 0.12, 17), np.linspace(0.2, 0.8, 25), indexing='ij')
    n1, n2 = state_space[0].shape
    loss_grid = np.zeros((n1, n2))
    grad_grid = [[None for _ in range(n2)] for _ in range(n1)]
    # Run sim on different mass state and compute loss
    for i in range(n1):
        for j in range(n2):
            mass = state_space[0][i, j] * np.ones(sim.block_object.num_particle)
            friction = state_space[1][i, j] * np.ones(sim.block_object.num_particle)
            sim.input_parameters(mass, mapping, friction, mapping, u)

            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEPS)

                for idx in loss_steps:
                    sim.compute_loss(idx, 
                                    sim_gt.body_qpos[idx][0],
                                    sim_gt.body_qpos[idx][1], 
                                    sim_gt.body_rpos[idx])

            print('mass: %4f, friction %4f, loss: %.9f, dl/dx: %.5f, %.5f'%(mass[0], friction[0],
                    sim.loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_friction.grad.to_numpy()[0]))

            # grad = sim.composite_mass.grad.to_numpy()

            loss_grid[i][j] = sim.loss[None]
            # grad_grid[i][j] = grad[:3]


    # fig = make_subplots(rows=4, cols=4, subplot_titles=("m3 = 0.05", "m3 = 0.06", "m3 = 0.07", "m3 = 0.08", 
    #         "m3 = 0.09", "m3 = 0.1", "m3 = 0.11", "m3 = 0.12", "m3 = 0.13", "m3 = 0.14", "m3 = 0.15", "m3 = 0.16",
    #         "m3 = 0.17", "m3 = 0.18", "m3 = 0.19", "m3 = 0.2"))
    # fig.add_trace(go.Contour(z=loss_grid[0], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),1,1)
    # fig.add_trace(go.Contour(z=loss_grid[1], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),1,2)
    # fig.add_trace(go.Contour(z=loss_grid[2], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),1,3)
    # fig.add_trace(go.Contour(z=loss_grid[3], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),1,4)
    # fig.add_trace(go.Contour(z=loss_grid[4], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),2,1)
    # fig.add_trace(go.Contour(z=loss_grid[5], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),2,2)
    # fig.add_trace(go.Contour(z=loss_grid[6], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),2,3)
    # fig.add_trace(go.Contour(z=loss_grid[7], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),2,4)
    # fig.add_trace(go.Contour(z=loss_grid[8], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),3,1)
    # fig.add_trace(go.Contour(z=loss_grid[9], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),3,2)
    # fig.add_trace(go.Contour(z=loss_grid[10], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),3,3)
    # fig.add_trace(go.Contour(z=loss_grid[11], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),3,4)
    # fig.add_trace(go.Contour(z=loss_grid[12], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),4,1)
    # fig.add_trace(go.Contour(z=loss_grid[13], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),4,2)
    # fig.add_trace(go.Contour(z=loss_grid[14], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),4,3)
    # fig.add_trace(go.Contour(z=loss_grid[15], dx=0.01, x0=0.05, dy=0.01, y0=0.05, contours=dict(start=0, end=8, size=0.2,)),4,4)
    # Plot the loss contour
    fig = go.Figure(data =
    go.Contour(
        z=loss_grid,
        y0=0.08,
        dy=0.0025,
        x0=0.2,
        dx=0.025,
        contours=dict(
            start=0,
            end=4500,
            size=30,
        ),
    ))
    fig.add_trace(go.Scatter(x=[0.3, 0.4, 0.5], y=[0.09, 0.1, 0.11]))
    fig.show()
