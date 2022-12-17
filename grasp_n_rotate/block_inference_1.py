'''
Block object system identification.
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
from taichi_pushing.optimizer.optim import Backtracking, BacktrackingMomentum
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 5
NUM_ITER = 100
SIM_STEPS = 50

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
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)

    # map mass and friction to regions 
    mapping = np.zeros(sim.ngeom)

    # Control input
    u = {4: [[10, 0, 5] for _ in range(60)],
         40: [[0, -10, -5] for _ in range(60)],
         15: [[-10, 0, 5] for _ in range(60)]}

    # Time steps to include in loss computation
    loss_steps = [SIM_STEPS-1]

    sim_gt.input_parameters(mass_gt, mapping, friction_gt, mapping, u, loss_steps)
    
    # Ground truth forward simulation
    sim_gt.run(SIM_STEPS, render=True)
    geom_pos_gt = sim_gt.geom_pos.to_numpy()

    trajectories = []
    for ep in range(NUM_EPOCH):

        mass = 0.1 + 0.1 * np.random.rand(sim.block_object.num_particle)
        # friction = 0.5 + 0.5 * np.random.rand(sim.block_object.num_particle)
        friction = friction_gt.copy()
        sim.input_parameters(mass, mapping, friction, mapping, u, loss_steps)

        # x = {"mass": mass, "friction": friction}
        # bounds = {"mass": [0.005, 0.3], "friction": [0.05, 1]}
        x = {"mass": mass}
        bounds = {"mass": [0.005, 0.3]}
        optim = BacktrackingMomentum(x, bounds, geom_pos_gt, momentum=0.5)

        trajectory = []
        for iter in range(NUM_ITER):
            trajectory.append(mass[0])

            ti.ad.clear_all_gradients()
            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEPS)
                sim.compute_loss(geom_pos_gt)

            print('Ep %4d, Iter %05d, loss: %.9f, dl/dm: %.5f, at m_0: %.5f, dl/dmu: %.5f, at mu_0: %.5f '%(
                    ep, iter, sim.loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_mass.to_numpy()[0], sim.composite_friction.grad.to_numpy()[0],
                    sim.composite_friction.to_numpy()[0]))

            # grads = {"mass": sim.composite_mass.grad.to_numpy(),
            #          "friction": sim.composite_friction.grad.to_numpy()}
            grads = {"mass": sim.composite_mass.grad.to_numpy()}                     
            x = optim.step(sim, grads)

            sim.input_parameters(x["mass"], mapping, friction, mapping, u, loss_steps)

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
