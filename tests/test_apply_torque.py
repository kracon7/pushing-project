import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.composite_util import Composite2D
from taichi_pushing.physics.pushing_simulator import PushingSimulator

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':
    composite = Composite2D(0)
    sim = PushingSimulator(composite)
    
    mass = np.ones(*composite.mass_dist.shape)
    mass[:4] = np.array([1, 1.5, 2, 2.5]) 
    sim.composite_mass.from_numpy(mass)

    # map mass to THREE regions 
    mapping = np.zeros(sim.num_particle)
    mapping[int(sim.num_particle/4):] += 1
    mapping[int(2*sim.num_particle/4):] += 1
    mapping[int(3*sim.num_particle/4):] += 1

    sim.mass_mapping.from_numpy(mapping)
    
    sim.clear_all()
    sim.hand_vel = -10

    sim.initialize(5)
    for s in range(sim.max_step-1):
        sim.collide(s)
        sim.apply_external(s, 50, 0, 0, 500000)
        sim.compute_ft(s)
        sim.update(s)
        sim.render(s)
