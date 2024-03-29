'''
Hidden state si loss contour and gradient direction plot with hidden state simulator
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
    mapping[sim.ngeom//2:] = 1

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEP-1, 1)]
    # loss_steps = [SIM_STEPS-1]

    # ===========  Solve for constraint forces  ============= #
    constraint_solver = ConstraintForceSolver(param_file, SIM_STEP)
    
    force = np.zeros((sim.ngeom, SIM_STEP, 2))
    torque = np.zeros((sim.ngeom, SIM_STEP))

    batch = [0, 53]
    torque[0], torque[53] = 5, 5
    # torque += np.random.uniform(size=(sim.ngeom, SIM_STEP))

    constraint_solver.input_parameters(mass_gt, mapping, friction_gt, mapping, force, torque)

    optim = Momentum(force, lr=300, bounds=[-100, 100], momentum=0.9, alpha=0.9)

    for i in range(1500):
        with ti.ad.Tape(constraint_solver.loss):
            constraint_solver.run(batch)
            constraint_solver.compute_loss(batch)

        grad = constraint_solver.external_force.grad.to_numpy()
        force = optim.step(grad)
        constraint_solver.update_parameter(force, "force")
        print("Iteration: %d, Loss: %.10f"%(i, constraint_solver.loss[None]))

        if i%300 == 0:
            constraint_solver.run(batch, render=True)

    force_torque = np.concatenate([constraint_solver.external_force.to_numpy(),
                    np.expand_dims(torque, axis=-1)], axis=2).tolist()
    u = {b: force_torque[b] for b in batch}

    # ===========  Run hidden state sim  ============= #
    hidden_state_gt = sim.map_to_hidden_state(mass_gt, mapping, friction_gt, mapping)
    sim_gt.input_parameters(hidden_state_gt["body_mass"], hidden_state_gt["body_inertia"], 
                hidden_state_gt["body_com"], hidden_state_gt["si"], mapping, u, loss_steps)
    sim_gt.run(SIM_STEP, render=True)
    body_qvel_gt = sim_gt.body_qvel.to_numpy()
    body_rvel_gt = sim_gt.body_rvel.to_numpy()

    # ===========  Hidden state inference mass parameters  ============= #
    hidden_state = hidden_state_gt.copy()
    
    l1 = [0.39, 0.59, 41]
    l2 = [0.39, 0.59, 41]
    state_space = np.meshgrid(np.linspace(*l1), np.linspace(*l2), indexing='ij')
    n1, n2 = state_space[0].shape
    loss_grid = np.zeros((n1, n2))
    grad_grid = np.zeros((n1, n2, 2))
    # Run sim on different mass state and compute loss
    for i in range(n1):
        for j in range(n2):
            hidden_state['si'][0], hidden_state["si"][1] = state_space[0][i,j], state_space[1][i,j]

            sim.input_parameters(hidden_state["body_mass"], hidden_state["body_inertia"], 
                        hidden_state["body_com"], hidden_state["si"], mapping, u, loss_steps)

            with ti.ad.Tape(sim.loss):
                sim.run(SIM_STEP)
                sim.compute_loss(body_qvel_gt, body_rvel_gt)

            print('Loss: %.9f, grad: %.5f, %.5f, at : %.5f, %.5f'%(
                    sim.loss[None], sim.composite_si.grad[0], sim.composite_si.grad[1], 
                    sim.composite_si[0], sim.composite_si[1]))

            grad = [sim.composite_si.grad.to_numpy()[0], 
                    sim.composite_si.grad.to_numpy()[1]]

            loss_grid[i, j] = sim.loss[None]
            grad_grid[i, j] = - np.array(grad) / np.linalg.norm(grad)
    
    f = ff.create_quiver(state_space[1], state_space[0], grad_grid[:,:,1], grad_grid[:,:,0],
                        scale=0.8 * (l1[1] - l1[0])/l1[2])
    trace1 = f.data[0]
    trace2 = go.Contour(z=loss_grid, y0=l1[0], dy=(l1[1] - l1[0])/l1[2] , x0=l2[0], dx=(l2[1] - l2[0])/l2[2],
                            contours=dict(
                                start=0,
                                end=loss_grid.max(),
                                size=loss_grid.max() / 100,
                            )
                        )
    # Plot the loss contour
    fig = go.Figure(data = [trace1, trace2])
    width = 1000
    height = width * (l1[1] - l1[0]) / (l2[1] - l2[0])
    fig.update_layout(title="hidden state si loss contour",
                      width = width,
                      height = height)
    fig.show()
