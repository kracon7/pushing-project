import os
import sys
import argparse
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping
from taichi_pushing.physics.constant_speed_constraint_solver import ConstantSpeedConstraintSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 5
NUM_ITER = 1000
SIM_STEP = 50

ti.init(arch=ti.cpu, debug=True)
np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = HiddenStateSimulator(param_file)

    mass = 0.1 * np.ones(sim.block_object.num_particle)
    friction = 0.5 * np.ones(sim.block_object.num_particle)
    n_partition = 6
    mass_mapping = np.repeat(np.arange(n_partition), 
                             sim.ngeom//n_partition).astype("int")
    friction_mapping = np.zeros(sim.block_object.num_particle).astype("int")
    assert mass_mapping.shape[0] == sim.ngeom

    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                    friction, friction_mapping)

    # ===========  Solve for constraint forces  ============= #
    constraint_solver = ConstantSpeedConstraintSolver(param_file, SIM_STEP)
    
    external_force = np.zeros((sim.ngeom, SIM_STEP, 2))
    external_torque = np.zeros(sim.ngeom)
    constraint_solver.external_force.from_numpy(external_force)
    constraint_solver.external_torque.from_numpy(external_torque)

    batch_speed = {i: 10 for i in range(sim.ngeom)}

    optim_f = Momentum(external_force, lr=30, bounds=[-80, 80], momentum=0.9, alpha=0.9)
    optim_t = Momentum(external_torque, lr=1e-2, bounds=[-5, 5], momentum=0.9, alpha=0.9)

    constraint_solver.input_parameters(hidden_state)
    constraint_solver.run(batch_speed, auto_diff=False, render=True)

    for i in range(3000):
        constraint_solver.run(batch_speed, auto_diff=True)

        grad_f = constraint_solver.external_force.grad.to_numpy()
        grad_t = constraint_solver.external_torque.grad.to_numpy()
        external_force = optim_f.step(grad_f)
        external_torque = optim_t.step(grad_t)
        constraint_solver.external_force.from_numpy(external_force)
        constraint_solver.external_torque.from_numpy(external_torque)

        print("Iteration: %d, Loss: %.10f"%(i, constraint_solver.loss[None]))
        if i%50 == 0:
            constraint_solver.run(batch_speed, auto_diff=False, render=True)

