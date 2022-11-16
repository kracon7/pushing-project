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
    
    mass = 0.1*np.ones(sim.block_object.num_particle)
    sim.composite_mass.from_numpy(mass)
    friction = np.ones(sim.block_object.num_particle)
    sim.composite_friction.from_numpy(friction)

    mapping = np.zeros(sim.ngeom)
    sim.mass_mapping.from_numpy(mapping)
    sim.friction_mapping.from_numpy(mapping)
    
    # Test forward simulation
    sim.clear_all()
    sim.initialize()
    for s in range(10):
        sim.bottom_friction(s)
        sim.apply_external(s, 10, 0, 0, 100)
        sim.compute_ft(s)
        sim.forward_body(s)
        sim.forward_geom(s)
        sim.render(s)

    # Test auto-differentiation
    sim.clear_all()
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    loss[None] = 0

    @ti.kernel
    def compute_loss():
        loss[None] = sim.body_qpos[19].norm()**2 + sim.body_rpos[19]**2

    with ti.ad.Tape(loss):
        sim.initialize()
        for s in range(20):
            sim.bottom_friction(s)
            sim.apply_external(s, 10, 0, 0, 100)
            sim.compute_ft(s)
            sim.forward_body(s)
            sim.forward_geom(s)

        compute_loss()

    print('loss: ', loss[None], 
          " dl/dqx: ", sim.body_qpos.grad.to_numpy()[0],
          " dl/drx: ", sim.body_rpos.grad.to_numpy()[0],
          " at qpos: ", sim.body_qpos.to_numpy()[0],
          " rpos: ", sim.body_rpos.to_numpy()[0])