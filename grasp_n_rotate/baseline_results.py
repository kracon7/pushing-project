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
    geom_pos_gt = sim_gt.geom_pos.to_numpy()

    # ===========  Load Baseline Parameters  ============= #
    baseline_data = np.loadtxt(os.path.join(data_dir, "baseline_results.txt"))
    for i in range(5):
        mass, friction = baseline_data[2*i], baseline_data[2*i+1]
        hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                                friction, friction_mapping)
        sim.input_parameters(hidden_state, u, loss_steps)
        sim.run(SIM_STEP, batch_speed, auto_diff=False, render=True)
        geom_pos = sim.geom_pos.to_numpy()

        ### Baseline results
        nad = np.sum(np.abs(mass - mass_gt)) / np.sum(mass_gt)
        mpd = np.average(np.linalg.norm(geom_pos[:,SIM_STEP-1,:] - geom_pos_gt[:,SIM_STEP-1,:], axis=-1))

        print("NAD: %.5f, MPD: %.5f"%(nad, mpd))