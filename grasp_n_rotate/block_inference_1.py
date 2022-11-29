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
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 5
NUM_ITER = 100
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
    u = {4: [[2, 0, 1] for _ in range(400)]}

    sim_gt.input_parameters(mass_gt, mapping, friction_gt, mapping, u)
    
    # Ground truth forward simulation
    sim_gt.run(SIM_STEPS)

    # Run system id
    loss = ti.field(ti.f64, shape=(), needs_grad=True)

    @ti.kernel
    def compute_loss(idx: ti.i64):
        loss[None] = 10 * (sim.body_qpos[4, idx] - sim_gt.body_qpos[4, idx]).norm()**2 + \
                     (sim.body_rpos[4, idx] - sim_gt.body_rpos[4, idx])**2

    trajectories = []
    for ep in range(NUM_EPOCH):

        mass = 0.1 + 0.1 * np.random.rand(sim.block_object.num_particle)
        friction = 0.5 + 0.5 * np.random.rand(sim.block_object.num_particle)
        sim.input_parameters(mass, mapping, friction, mapping, u)

        trajectory = []
        for iter in range(NUM_ITER):
            trajectory.append(mass[0])

            with ti.ad.Tape(loss):
                sim.run(SIM_STEPS)
                compute_loss(SIM_STEPS - 1)

            print('Ep %4d, Iter %05d, loss: %.9f, dl/dm: %.5f, at m_0: %.5f, dl/dmu: %.5f, at mu_0: %.5f '%(
                    ep, iter, loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_mass.to_numpy()[0], sim.composite_friction.grad.to_numpy()[0],
                    sim.composite_friction.to_numpy()[0]))

            mass_grad = sim.composite_mass.grad
            friction_grad = sim.composite_friction.grad
            mass -= 0.000003 * mass_grad.to_numpy()
            friction -= 0.00003 * friction_grad.to_numpy()
            sim.input_parameters(mass, mapping, friction, mapping, u)

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
