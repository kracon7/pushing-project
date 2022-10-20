import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    
    mass = np.ones(sim.block_object.num_particle)
    # mass[:4] = np.array([1, 1.5, 2, 2.5]) 
    sim.composite_mass.from_numpy(mass)

    # # map mass to THREE regions 
    mapping = np.zeros(sim.ngeom)
    # mapping[int(sim.num_particle/4):] += 1
    # mapping[int(2*sim.num_particle/4):] += 1
    # mapping[int(3*sim.num_particle/4):] += 1

    sim.mass_mapping.from_numpy(mapping)
    
    sim.clear_all()

    sim.initialize()
    for s in range(sim.max_step-1):
        sim.bottom_friction(s)
        sim.apply_external(s, 10, 0, 0, 100)
        sim.compute_ft(s)
        sim.forward_body(s)
        sim.forward_geom(s)
        sim.render(s)
