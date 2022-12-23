'''
Test hidden state inferencing with hidden state simulator
'''

import os
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator
from taichi_pushing.physics.constraint_force_solver import ConstraintForceSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

SIM_STEP = 50

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = HiddenStateSimulator(param_file)
    sim_gt = HiddenStateSimulator(param_file)

    mass_gt = 0.1 * np.ones(sim.block_object.num_particle)
    friction_gt = 0.5 * np.ones(sim.block_object.num_particle)
    mapping = np.zeros(sim.ngeom).astype("int")
    hidden_state_gt = sim.map_to_hidden_state(mass_gt, mapping, friction_gt, mapping)

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEP-1, 1)]
    # loss_steps = [SIM_STEPS-1]

    # ===========  Solve for constraint forces  ============= #
    constraint_solver = ConstraintForceSolver(param_file, SIM_STEP)
    
    force = np.zeros((sim.ngeom, SIM_STEP, 2))
    torque = np.zeros((sim.ngeom, SIM_STEP))

    batch = [0, 20, 53]
    torque[0], torque[20], torque[53] = 5, 3, 5
    torque += np.random.uniform(size=(sim.ngeom, SIM_STEP))

    constraint_solver.input_parameters(mass_gt, mapping, friction_gt, mapping, force, torque)

    optim = Momentum(force, lr=200, bounds=[-100, 100], momentum=0.9)

    for i in range(2):
        with ti.ad.Tape(constraint_solver.loss):
            constraint_solver.run(batch)
            constraint_solver.compute_loss(batch)

        grad = constraint_solver.external_force.grad.to_numpy()
        force = optim.step(grad)
        constraint_solver.update_parameter(force, "force")
        print("Iteration: %d, Loss: %.10f"%(i, constraint_solver.loss[None]))

        if i%100 == 0:
            constraint_solver.run(batch, render=True)

    force_torque = np.concatenate([constraint_solver.external_force.to_numpy(),
                    np.expand_dims(torque, axis=-1)], axis=2).tolist()
    u = {b: force_torque[b] for b in batch}

    # ===========  Run hidden state sim  ============= #
    sim_gt.input_parameters(hidden_state_gt["body_mass"], hidden_state_gt["body_inertia"], 
                hidden_state_gt["body_com"], hidden_state_gt["si"], mapping, u, loss_steps)
    sim_gt.run(SIM_STEP, render=True)
    body_qvel_gt = sim_gt.body_qvel.to_numpy()
    body_rvel_gt = sim_gt.body_rvel.to_numpy()

    # ===========  Hidden state inference mass parameters  ============= #
    hidden_state = hidden_state_gt.copy()
    hidden_state["body_mass"] += 0.3 * hidden_state["body_mass"]
    hidden_state["body_inertia"] -= 0.3 * hidden_state["body_inertia"]
    hidden_state["body_com"] += 0.3 * hidden_state["body_com"]

    sim.input_parameters(hidden_state["body_mass"], hidden_state["body_inertia"], 
                hidden_state["body_com"], hidden_state["si"], mapping, u, loss_steps)

    for i in range(1000):
        with ti.ad.Tape(sim.loss):
            sim.run(SIM_STEP)
            sim.compute_loss(body_qvel_gt, body_rvel_gt)

        print('Iter %05d, loss: %.9f, grad: %.5f, %.5f, %.5f, %.5f, %.5f,'
              ' at : %.5f, %.5f, %.5f, %.5f, %.5f'%(i, sim.loss[None], 
              sim.body_mass.grad[None], sim.body_inertia.grad[None], sim.body_com.grad[None][0],
              sim.body_com.grad[None][1], sim.composite_si.grad[0], 
              sim.body_mass[None], sim.body_inertia[None], sim.body_com[None][0],
              sim.body_com[None][1], sim.composite_si[0]))