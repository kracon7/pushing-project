'''
Hidden state si loss contour and gradient direction plot with hidden state simulator
'''

import os
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import taichi as ti
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping
from taichi_pushing.physics.constraint_force_solver import ConstraintForceSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults

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
    n_partition = 6
    mass_mapping = np.repeat(np.arange(n_partition), 
                             sim.ngeom//n_partition).astype("int")
    friction_mapping = np.zeros(sim.block_object.num_particle).astype("int")
    assert mass_mapping.shape[0] == sim.ngeom

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEP-1, 1)]
    # loss_steps = [SIM_STEPS-1]

    # ===========  Solve for constraint forces  ============= #
    constraint_solver = ConstraintForceSolver(param_file, SIM_STEP)
    
    force = np.zeros((sim.ngeom, SIM_STEP, 2))
    torque = np.zeros((sim.ngeom, SIM_STEP))

    batch = [i * (sim.ngeom//n_partition) for i in range(n_partition)]
    for b in batch:
        torque[b] = 4

    constraint_solver.input_parameters(mass_gt, mass_mapping, 
                                    friction_gt, friction_mapping, force, torque)

    optim = Momentum(force, lr=300, bounds=[-100, 100], momentum=0.9, alpha=0.9)

    for i in range(10):
        with ti.ad.Tape(constraint_solver.loss):
            constraint_solver.run(batch)
            constraint_solver.compute_loss(batch)

        grad = constraint_solver.external_force.grad.to_numpy()
        force = optim.step(grad)
        constraint_solver.update_parameter(force, "force")

        if i%300 == 0:
            print("Iteration: %d, Loss: %.10f"%(i, constraint_solver.loss[None]))
            constraint_solver.run(batch, render=True)

    force_torque = np.concatenate([constraint_solver.external_force.to_numpy(),
                    np.expand_dims(torque, axis=-1)], axis=2).tolist()
    u = {b: force_torque[b] for b in batch}

    # ===========  Run the ground truth hidden state sim  ============= #
    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state_gt = hidden_state_mapping.map_to_hidden_state(mass_gt, mass_mapping,
                                                    friction_gt, friction_mapping)
    sim_gt.input_parameters(hidden_state_gt, u, loss_steps)
    sim_gt.run(SIM_STEP, render=True)
    body_qvel_gt = sim_gt.body_qvel.to_numpy()
    body_rvel_gt = sim_gt.body_rvel.to_numpy()

    # ===========  Hidden state inference mass parameters  ============= #
    hidden_state = hidden_state_gt.copy()
    for i in range(n_partition):
        hidden_state['composite_si'][i] = 0.39 + 0.1 * np.random.randint(3)

    sim.input_parameters(hidden_state, u, loss_steps)

    # Add parameter to the optimizer
    si_optim = Momentum(hidden_state['composite_si'], lr=1e-1, bounds=[0, 1]
    )
    # GD with momentum
    for i in range(500):
        with ti.ad.Tape(sim.loss):
            sim.run(SIM_STEP)
            sim.compute_loss(body_qvel_gt, body_rvel_gt)

        grad = sim.composite_si.grad.to_numpy()
        hidden_state['composite_si'] = si_optim.step(grad)
        sim.update_parameter("composite_si", hidden_state["composite_si"])

        print('Iteration: %d, Loss: %.9f'%(i, sim.loss[None]), hidden_state["composite_si"][:n_partition])
