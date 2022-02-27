'''
Test for PushingSimulator with contact and bottom friction
Object assumed to have THREE regions of mass distribution through mass mapping
'''

import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.composite_util import Composite2D
from taichi_pushing.physics.pushing_simulator import PushingSimulator
from taichi_pushing.physics.utils import Defaults

import matplotlib.pyplot as plt

np.random.seed(0)

DTYPE = Defaults.DTYPE

ti.init(arch=ti.cpu, debug=False)

@ti.data_oriented
class Loss():
    """docstring for Loss"""
    def __init__(self, engines):
        self.loss = ti.field(DTYPE, shape=(), needs_grad=True)
        self.engines = engines

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.kernel
    def compute_loss(self, t: ti.i32):
        for i in sim_est.composite_geom_id:
            self.loss[None] += (self.engines[0].geom_pos[t, i][0] - self.engines[1].geom_pos[t, i][0])**2 + \
                    (self.engines[0].geom_pos[t, i][1] - self.engines[1].geom_pos[t, i][1])**2



def run_world(sim, action_idx):
    sim.initialize(action_idx)
    for s in range(sim.max_step-1):
        sim.collide(s)
        sim.compute_ft(s)
        sim.update(s)
        # sim.render(s)

def run_episode(action_idx=10):
    run_world(sim_est, action_idx)
    run_world(sim_gt, action_idx)

def forward():
    run_episode()
    loss.compute_loss(400)

obj_idx = 0
composite_est, composite_gt = Composite2D(obj_idx), Composite2D(obj_idx)
sim_est, sim_gt = PushingSimulator(composite_est), PushingSimulator(composite_gt)
mass_gt = np.ones(*composite_est.mass_dist.shape)
mass_gt[:3] = np.array([1, 1.5, 2]) 
sim_gt.composite_mass.from_numpy(mass_gt)

# map mass to THREE regions 
mapping = np.zeros(sim_gt.num_particle)
mapping[int(sim_gt.num_particle/3):] += 1
mapping[int(2*sim_gt.num_particle/3):] += 1

sim_gt.mass_mapping.from_numpy(mapping)
sim_est.mass_mapping.from_numpy(mapping)

loss = Loss((sim_est, sim_gt))

lr = 5e-3
max_iter = 120

traj = []
h = 5
for k in range(h):

    mass_est = 4*np.random.rand(*composite_est.mass_dist.shape)
    sim_est.composite_mass.from_numpy(mass_est)

    temp = [mass_est[:3].copy()]

    for i in range(max_iter):
        sim_gt.clear_all()
        sim_est.clear_all()
        loss.clear_loss()

        # forward sims to compute loss and gradient
        with ti.Tape(loss.loss):
            forward()

        print('Iteration %d loss: %12.4f  gt mass:  %16.8f  estimated mass: %16.8f  gradient: %.4f'%(
                i, loss.loss[None], mass_gt[0], sim_est.composite_mass[0], sim_est.composite_mass.grad[0]))

        grad = sim_est.composite_mass.grad

        # update estimated mass
        mass_est -= lr * grad.to_numpy()
        sim_est.composite_mass.from_numpy(mass_est)

        temp.append(mass_est[:4].copy())

    traj.append(np.stack(temp))


c = np.array([[207, 40, 30],
              [255, 143, 133],
              [102, 0, 162],
              [72, 0, 141],
              [0, 162, 132],
              [90, 228, 165],
              [241, 255, 74],
              [240, 225, 0]]).astype('float') / 255

fig, ax = plt.subplots(1,1)
ax.plot(mass_gt[0]*np.ones(max_iter), color=c[0], linestyle='dashed')
ax.plot(mass_gt[1]*np.ones(max_iter), color=c[2], linestyle='dashed')
ax.plot(mass_gt[2]*np.ones(max_iter), color=c[4], linestyle='dashed')
for i in range(h):
    ax.plot(traj[i][:,0], color=c[1], alpha=0.5)
    ax.plot(traj[i][:,1], color=c[3], alpha=0.5)
    ax.plot(traj[i][:,2], color=c[5], alpha=0.5)

plt.show()
