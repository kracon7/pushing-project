import os
import sys
import argparse
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.constraint_force_solver import ConstraintForceSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

NUM_EPOCH = 5
NUM_ITER = 200
SIM_STEP = 50

ti.init(arch=ti.cpu, debug=True)
np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Solve constraint forces for pure rotation')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    constraint_solver = ConstraintForceSolver(param_file, SIM_STEP)
    
    mass = 0.1 * np.ones(sim.block_object.num_particle)
    friction = 0.5 * np.ones(sim.block_object.num_particle)
    mapping = np.zeros(sim.ngeom)

    force = np.zeros((sim.ngeom, SIM_STEP, 2))
    torque = np.zeros((sim.ngeom, SIM_STEP))

    batch = [0,  53]
    for b in batch:
        torque[b] = 2

    constraint_solver.input_parameters(mass, mapping, friction, mapping, force, torque)

    optim = Momentum(force, lr=1000000, bounds=[-100, 100], momentum=0.8)

    for i in range(NUM_ITER):
        with ti.ad.Tape(constraint_solver.loss):
            constraint_solver.run(batch)
            constraint_solver.compute_loss(batch)

        grad = constraint_solver.external_force.grad.to_numpy()
        force = optim.step(grad)
        constraint_solver.update_parameter(force, "force")
        print("Iteration: %d, Loss: %.8f"%(i, constraint_solver.loss[None]))

        if i%30 == 0:
            constraint_solver.run(batch, render=True)

    force_torque = np.concatenate([constraint_solver.external_force.to_numpy(),
                    np.expand_dims(torque, axis=-1)], axis=2).tolist()
    u = {b: force_torque[b] for b in batch}

    loss_steps = [SIM_STEP-1]

    sim.input_parameters(mass, mapping, friction, mapping, u, loss_steps)
    sim.run(SIM_STEP, render=True)
