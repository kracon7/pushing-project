import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator

ti.init(arch=ti.cpu, debug=True)

SIM_STEPS = 400

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    
    mass = 0.1*np.ones(sim.block_object.num_particle)
    friction = np.ones(sim.block_object.num_particle)
    mapping = np.zeros(sim.ngeom)
    u = [[4, 5, 0, 1] for _ in range(SIM_STEPS)]

    sim.input_parameters(mass, mapping, friction, mapping, u)
    
    # Test forward simulation
    sim.run(SIM_STEPS)

    # Test auto-differentiation
    sim.clear_all()
    loss = ti.field(ti.f64, shape=(), needs_grad=True)
    loss[None] = 0

    @ti.kernel
    def compute_loss():
        loss[None] = 10 * (sim.body_qpos[19].norm()**2 + sim.body_rpos[19]**2)

    with ti.ad.Tape(loss):
        sim.run(20)
        compute_loss()

    print('loss: %.9f, dl/dm: %.5f, at m_0: %.5f, dl/dmu: %.5f, at mu_0: %.5f '%(
           loss[None], sim.composite_mass.grad.to_numpy()[0], sim.composite_mass.to_numpy()[0], 
           sim.composite_friction.grad.to_numpy()[0], sim.composite_friction.to_numpy()[0]))