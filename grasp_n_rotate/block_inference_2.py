'''
Block object system identification. Using GD + Backtracking line search
Run ground truch simulation for once, then random initialize sim estimation 
for NUM_EPOCH times.

Assumptions:
    -- uniform friction profile
    -- non-uniform mass
'''

import os
import sys
import argparse
import numpy as np
import taichi as ti
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt
from math import sin, cos

NUM_EPOCH = 1
NUM_ITER = 200
SIM_STEPS = 50
LR = 0.01

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    sim_gt = GraspNRotateSimulator(param_file)
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)
    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)

    # # map mass to regions 
    mapping = sim.block_object.mass_mapping
    mapping[:27] = 0
    mapping[27:] = 1
    num_region = mapping.max() + 1     # number of regions in particle mass mapping

    # Control input
    # u = [[4, 3, 0, 1] for _ in range(100)] + \
    #     [[40, 0, 3, 0.5] for _ in range(200)] + \
    #     [[10, 3.7, 0, 0.1] for _ in range(200)]
    u = [[4, 6 * sin(i/100), 6 * cos(i/100), 1] for i in range(500)]

    sim_gt.input_parameters(mass_gt, mapping, friction_gt, mapping, u)

    # Time steps to include in loss computation
    loss_steps = [SIM_STEPS-1]
        
    # Ground truth forward simulation
    sim_gt.run(SIM_STEPS, render=True)

    # Compute and plot the loss contour
    state_space = np.meshgrid(np.linspace(0.08, 0.13, 26), np.linspace(0.08, 0.13, 26), indexing='ij')
    n1, n2 = state_space[0].shape
    loss_grid = np.zeros((n1, n2))
    grad_grid = [[None for _ in range(n2)] for _ in range(n1)]
    # Run sim on different mass state and compute loss
    for i in range(n1):
        for j in range(n2):
            mass = np.ones(sim.block_object.num_particle)
            mass[0] = state_space[0][i, j]
            mass[1] = state_space[1][i, j]
            sim.input_parameters(mass, mapping, friction_gt, mapping, u)

            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEPS)

                for idx in loss_steps:
                    sim.compute_loss(idx, 
                                    sim_gt.body_qpos[idx][0],
                                    sim_gt.body_qpos[idx][1], 
                                    sim_gt.body_rpos[idx])
            print("i: %d, j: %d, mass: %.4f, %.4f, loss: %.4f"%(i, j, 
                    sim.composite_mass.to_numpy()[0], sim.composite_mass.to_numpy()[1],
                    sim.loss[None]))
            loss_grid[i][j] = sim.loss[None]

    # Run system id
    trajectories = []
    for ep in range(NUM_EPOCH):
        
        # Random initialization of mass estimation
        mass = 0.1 + 0.1 * np.random.rand(sim.block_object.num_particle)
        mass[0], mass[1] = 0.11, 0.125
        sim.input_parameters(mass, mapping, friction_gt, mapping, u)

        trajectory = []
        for iter in range(NUM_ITER):
            trajectory.append(mass[:num_region].copy())

            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEPS)

                for idx in loss_steps:
                    sim.compute_loss(idx, 
                                     sim_gt.body_qpos[idx][0],
                                     sim_gt.body_qpos[idx][1], 
                                     sim_gt.body_rpos[idx])

            print('Ep %4d, Iter %05d, loss: %.9f, dl/dm: %.5f  %.5f, at m: %.5f  %.5f'%(
                    ep, iter, sim.loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_mass.grad.to_numpy()[1], sim.composite_mass.to_numpy()[0],
                    sim.composite_mass.to_numpy()[1]))

            grad = sim.composite_mass.grad.to_numpy()

            lr = sim.backtracking(mass, grad, SIM_STEPS, sim_gt, u, loss_steps, 
                                lr_0=0.0001, alpha=0.5, lr_min=1e-7)
            mass -= lr * grad
            sim.composite_mass.from_numpy(mass)

            if abs(sim.loss[None]) < 1e-5:
                break

        trajectories.append(np.stack(trajectory))

    fig = go.Figure(data =
    go.Contour(
        z=loss_grid,
        y0=0.08,
        dy=0.002,
        x0=0.08,
        dx=0.002,
        contours=dict(
            start=0,
            end=5700,
            size=20,
        ),
    ))
    for i in range(NUM_EPOCH):
        fig.add_trace(go.Scatter(x=trajectories[i][:, 0], y=trajectories[i][:, 1]))
    fig.show()


    # print("Final regression results:")
    # for trajectory in trajectories:
    #     print(trajectory[-1])

    # # plot the trajectories
    # c = np.array([[207, 20, 20],
    #               [255, 143, 133],
    #               [102, 0, 162],
    #               [0, 162, 132],
    #               [90, 228, 165],
    #               [241, 255, 74],
    #               [255, 125, 0]]).astype('float') / 255

    # fig, ax = plt.subplots(1,1)
    # ax.plot(mass_gt[0]*np.ones(NUM_ITER), color=c[0], linestyle='dashed')
    # for i in range(NUM_EPOCH):
    #     for j in range(num_region):
    #         ax.plot(trajectories[i][:, j], color=c[j+1], alpha=0.6)

    # plt.show()
