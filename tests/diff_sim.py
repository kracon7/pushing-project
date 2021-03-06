import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.composite_util import Composite2D
from taichi_pushing.physics.pushing_simulator import PushingSimulator
from taichi_pushing.physics.utils import Defaults

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

obj_idx = 0
composite_est, composite_gt = Composite2D(obj_idx), Composite2D(obj_idx)
sim_est, sim_gt = PushingSimulator(composite_est), PushingSimulator(composite_gt)

sim_gt.composite_mass.from_numpy(composite_est.mass_dist)
sim_est.composite_mass.from_numpy(np.random.rand(*composite_est.mass_dist.shape))
sim_gt.clear_all()
sim_est.clear_all()

num_particle = composite_est.num_particle

loss = Loss((sim_est, sim_gt))

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
    loss.compute_loss(100)

with ti.Tape(loss.loss):
    forward()

# forward()

print(loss.loss[None])
print(sim_est.composite_mass.grad)