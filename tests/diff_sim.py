import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.composite_util import Composite2D
from taichi_pushing.pushing_simulator import PushingSimulator

ti.init(arch=gpu)

obj_idx = 0
composite_est, composite_gt = Composite2D(obj_idx), Composite2D(obj_idx)
sim_est, sim_gt = PushingSimulator(composite_est), PushingSimulator(composite_gt)

sim_est.mass = ti.field(ti.f64, shape=composite_est.mass_dim, needs_grad=True)

sim_gt.mass.from_numpy(composite_est.mass_dist)
sim_est.mass.from_numpy(np.random.rand(*composite_est.mass_dist.shape))

def run_world(sim, action_idx):
	sim.initialize(action_idx)
	for s in range(sim.max_step-1):
	    sim.collide(s)
	    sim.compute_ft(s)
	    sim.update(s)
	    sim.render(s)

def run_episode(action_idx=10):
	run_world(sim_est, action_idx)
	run_world(sim_gt, action_idx)

run_episode()