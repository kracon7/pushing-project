'''
Block object system identification.
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
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 4
NUM_ITER = 1000
SIM_STEPS = 200
LR = 0.01

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    sim_gt = GraspNRotateSimulator(param_file)
    
    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    sim_gt.composite_mass.from_numpy(mass_gt)

    # # map mass to regions 
    mapping = sim.block_object.mass_mapping
    # mapping[:18] = 0
    # mapping[18 : 36] = 1
    # mapping[36:] = 2
    num_region = mapping.max() + 1     # number of regions in particle mass mapping

    sim.mass_mapping.from_numpy(mapping)
    sim_gt.mass_mapping.from_numpy(mapping)
    
    # Ground truth forward simulation
    sim_gt.clear_all()
    sim_gt.initialize()
    for s in range(SIM_STEPS):
        sim_gt.bottom_friction(s)
        sim_gt.apply_external(s, 4, 5, 0, 1)
        sim_gt.compute_ft(s)
        sim_gt.forward_body(s)
        sim_gt.forward_geom(s)
        sim_gt.render(s)

    # Run system id
    loss = ti.field(ti.f64, shape=(), needs_grad=True)

    @ti.kernel
    def compute_loss(idx: ti.i64):
        loss[None] = (sim.body_qpos[idx] - sim_gt.body_qpos[idx]).norm()**2 + \
                     (sim.body_rpos[idx] - sim_gt.body_rpos[idx])**2

    trajectories = []
    for ep in range(NUM_EPOCH):
        
        # Random initialization of mass estimation
        mass = 0.07 + 0.1 * np.random.rand(sim.block_object.num_particle)
        sim.composite_mass.from_numpy(mass)

        trajectory = []
        for iter in range(NUM_ITER):
            trajectory.append(mass[:num_region].copy())
            loss[None] = 0
            loss.grad[None] = 0

            with ti.ad.Tape(loss):
                sim.clear_all()
                sim.initialize()
                for s in range(SIM_STEPS):
                    sim.bottom_friction(s)
                    sim.apply_external(s, 4, 5, 0, 1)
                    sim.compute_ft(s)
                    sim.forward_body(s)
                    sim.forward_geom(s)

                compute_loss(SIM_STEPS - 1)

            print('Ep %4d, Iter %05d, loss: %.9f, dl/dm: %.5f, at m_0: %.5f'%(
                    ep, iter, loss[None], sim.composite_mass.grad.to_numpy()[0],
                    sim.composite_mass.to_numpy()[0]))

            grad = sim.composite_mass.grad
            mass -= LR * grad.to_numpy()
            sim.composite_mass.from_numpy(mass)

        trajectories.append(np.stack(trajectory))

    print("Final regression results:")
    for trajectory in trajectories:
        print(trajectory[-1])

    # plot the trajectories
    c = np.array([[207, 20, 20],
                  [255, 143, 133],
                  [102, 0, 162],
                  [0, 162, 132],
                  [90, 228, 165],
                  [241, 255, 74],
                  [255, 125, 0]]).astype('float') / 255

    fig, ax = plt.subplots(1,1)
    ax.plot(mass_gt[0]*np.ones(NUM_ITER), color=c[0], linestyle='dashed')
    for i in range(NUM_EPOCH):
        for j in range(num_region):
            ax.plot(trajectories[i][:, j], color=c[j+1], alpha=0.6)

    plt.show()
