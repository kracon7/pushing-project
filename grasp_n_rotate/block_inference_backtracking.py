'''
Block object system identification using GD with Backtracking Line Search
Run ground truch simulation for once, then random initialize sim estimation 
for NUM_EPOCH times.

Assumptions:
    -- uniform friction profile
    -- uniform mass
'''

import os
import sys
import argparse
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 10
NUM_ITER = 20
SIM_STEPS = 400
LR = 0.1

ti.init(arch=ti.cpu, debug=True)
np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    sim_gt = GraspNRotateSimulator(param_file)
    
    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    sim_gt.composite_mass.from_numpy(mass_gt)

    # map mass to regions 
    mapping = np.zeros(sim.ngeom)

    sim.mass_mapping.from_numpy(mapping)
    sim_gt.mass_mapping.from_numpy(mapping)

    # Control input
    u = [[4, 5, 0, 1] for _ in range(SIM_STEPS)]
 
    # Time steps to include in loss computation
    loss_steps = [100, 200, SIM_STEPS-1]
       
    # Ground truth forward simulation
    sim_gt.clear_all()
    sim_gt.initialize()
    for s in range(SIM_STEPS):
        sim_gt.bottom_friction(s)
        sim_gt.apply_external(s, u[s][0], u[s][1], u[s][2], u[s][3])
        sim_gt.compute_ft(s)
        sim_gt.forward_body(s)
        sim_gt.forward_geom(s)
        sim_gt.render(s)

    # Run system id
    trajectories = []
    for ep in range(NUM_EPOCH):

        mass = 0.1 + 0.1 * np.random.rand(sim.block_object.num_particle)
        sim.composite_mass.from_numpy(mass)

        trajectory = []
        for iter in range(NUM_ITER):
            trajectory.append(mass[0])
            sim.loss[None] = 0
            sim.loss.grad[None] = 0

            with ti.ad.Tape(sim.loss):
                sim.clear_all()
                sim.initialize()
                for s in range(SIM_STEPS):
                    sim.bottom_friction(s)
                    sim.apply_external(s, u[s][0], u[s][1], u[s][2], u[s][3])
                    sim.compute_ft(s)
                    sim.forward_body(s)
                    sim.forward_geom(s)

                for idx in loss_steps:
                    sim.compute_loss(idx, 
                                    sim_gt.body_qpos[idx][0],
                                    sim_gt.body_qpos[idx][1], 
                                    sim_gt.body_rpos[idx])

            print('Ep %4d, Iter %05d, loss: %.9f, dl/dm: %.5f, at m_0: %.5f'%(
                    ep, iter, sim.loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_mass.to_numpy()[0]))

            grad = sim.composite_mass.grad.to_numpy()

            lr = sim.backtracking(mass, grad, SIM_STEPS, sim_gt, u, loss_steps)

            mass -= lr * grad
            sim.composite_mass.from_numpy(mass)

            if abs(sim.loss[None]) < 1e-5:
                break

        trajectories.append(trajectory)

    # plot the trajectories
    c = np.array([[207, 40, 30],
                [255, 143, 133],
                [102, 0, 162],
                [72, 0, 141],
                [0, 162, 132],
                [90, 228, 165],
                [241, 255, 74],
                [240, 225, 0]]).astype('float') / 255

    fig, ax = plt.subplots(1,1)
    ax.plot(mass_gt[0]*np.ones(NUM_ITER), color=c[0], linestyle='dashed')
    for i in range(NUM_EPOCH):
        ax.plot(trajectories[i], color=c[1], alpha=0.6)

    plt.show()
