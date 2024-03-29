'''
Hidden state si inference from recorded trajectories
Mapping back to explicit states
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

SIM_STEP = 100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    parser.add_argument("--data_dir", type=str, default='data/block_object_I_1')
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip('/')
    data_suffix = data_dir.split('/')[-1].split('_')[-1]
    param_file = data_dir.split('/')[-1].rstrip('_' + data_suffix) + '.yaml'
    print("Loading data batch %s for %s"%(data_suffix, param_file))
    param_file = os.path.join(ROOT, 'config', param_file)
    sim = HiddenStateSimulator(param_file)
    sim_gt = HiddenStateSimulator(param_file)

    # Load ground truth states
    mass_gt = np.loadtxt(os.path.join(data_dir, 'composite_mass.txt'))
    friction_gt = np.loadtxt(os.path.join(data_dir, 'composite_friction.txt'))
    mass_mapping = np.loadtxt(os.path.join(data_dir, 'mass_mapping.txt')).astype("int")
    friction_mapping = np.loadtxt(os.path.join(data_dir, 'friction_mapping.txt')).astype("int")
    assert mass_mapping.shape[0] == sim.ngeom

    # Load external perturbations
    u_idx = [i for i in range(sim.ngeom)]
    u = {}
    for i in u_idx:
        u[i] = np.loadtxt(os.path.join(data_dir,  "ft_%d.txt"%i))[:,1:]

    # Load batch speed
    batch_speed = {}
    temp = np.loadtxt(os.path.join(data_dir, 'batch_speed.txt'))
    for i in u_idx:
        batch_speed[i] = temp[i, 1]

    # Time steps to include in loss computation
    loss_steps = [i for i in range(2, SIM_STEP-1, 1)]
    # loss_steps = [SIM_STEPS-1]

    # ===========  Run the ground truth hidden state sim  ============= #
    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state_gt = hidden_state_mapping.map_to_hidden_state(mass_gt, mass_mapping,
                                                    friction_gt, friction_mapping)
    sim_gt.input_parameters(hidden_state_gt, u, loss_steps)
    sim_gt.run(SIM_STEP, batch_speed, auto_diff=False, render=True)
    body_qvel_gt = sim_gt.body_qvel.to_numpy()
    body_rvel_gt = sim_gt.body_rvel.to_numpy()

    # ===========  Hidden state inference mass parameters  ============= #
    hidden_state = hidden_state_gt.copy()
    # Add noise
    hidden_state["body_inertia"] += 0.05 * hidden_state["body_inertia"] 
    hidden_state["body_com"] -= 0.05 * hidden_state["body_com"]
    for i in range(sim.ngeom):
        hidden_state['composite_si'][i] = 1

    sim.input_parameters(hidden_state, u, loss_steps)

    # Add parameter to the optimizer
    si_optim = Momentum(hidden_state['composite_si'], lr=2e-1, bounds=[0.1, 5])

    f = open(os.path.join(data_dir, 'si_inference.txt'), 'w')
    
    # GD with momentum
    for i in range(2000):
        sim.input_parameters(hidden_state, u, loss_steps)
        sim.reset()
        sim.apply_initial_speed(batch_speed)
        
        with ti.ad.Tape(sim.loss):
            sim.initialize()
            for k in sim.u:
                for s in range(SIM_STEP):
                    sim.bottom_friction(k, s)
                    sim.apply_external(k, s, sim.u[k][s][0], sim.u[k][s][1], sim.u[k][s][2])
                    sim.compute_ft(k, s)
                    sim.forward_body(k, s)
                    sim.forward_geom(k, s+1)
        
            sim.compute_loss(body_qvel_gt, body_rvel_gt)

        grad = sim.composite_si.grad.to_numpy()
        hidden_state['composite_si'] = si_optim.step(grad)

        print('Iteration: %d, Loss: %.9f'%(i, sim.loss[None]), hidden_state["composite_si"])
        f.write('Iteration: %d, Loss: %.9f\n'%(i, sim.loss[None]))

        if (i+1) % 20 == 0:
            explicit_state = hidden_state_mapping.map_to_explicit_state(hidden_state, 
                                                                        mass_mapping, 
                                                                        friction_mapping)
            print("Composite mass: " + np.array_str(explicit_state["composite_mass"], precision=4))
            print("Composite friction: " + np.array_str(explicit_state["composite_friction"], precision=4))
            f.write("Composite mass: " + np.array_str(explicit_state["composite_mass"], precision=4) + '\n')
            f.write("Composite friction: " + np.array_str(explicit_state["composite_friction"], precision=4) + '\n')

    f.close()