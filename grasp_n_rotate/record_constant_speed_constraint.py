'''
Record the force/torque during a constant-speed rotation wrt. certain particles

Recorded data will be saved to directory named by ($param_file_name)_($SUFFIX)
Data logger will save the explicit states
Data logger will also log the force torque to text files every k iterations

Modify mass, friction, mass_mapping, friction mapping for different explicit states

Modify batch_speed to change the rotation center and rotation speed
Only particles existed in batch_speed.keys() will be loaded and optimized
'''
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

NUM_ITER = 1500
SIM_STEP = 100
SUFFIX = "1"

ti.init(arch=ti.cpu, debug=True)
np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    parser.add_argument("--param_file", type=str, default='block_object_param.yaml')
    parser.add_argument("--load_history", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=200)
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', args.param_file)
    sim = HiddenStateSimulator(param_file, max_step=SIM_STEP+2)

    mass = 0.4 * np.ones(sim.block_object.num_particle)
    # mass = np.array([0.4, 0.1, 0.1, 0.4, 0.4, 0.4])
    friction = 0.5 * np.ones(sim.block_object.num_particle)
    n_partition = 6
    mass_mapping = np.repeat(np.arange(n_partition), 
                             sim.ngeom//n_partition).astype("int")
    friction_mapping = np.zeros(sim.block_object.num_particle).astype("int")
    assert mass_mapping.shape[0] == sim.ngeom

    # Save explicit states
    data_dir = os.path.join(ROOT, 'data', args.param_file.replace(".yaml", "_") + SUFFIX)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    np.savetxt(os.path.join(data_dir, 'composite_mass.txt'), mass, fmt='%10.6f')
    np.savetxt(os.path.join(data_dir, 'mass_mapping.txt'), mass_mapping, fmt='%d')
    np.savetxt(os.path.join(data_dir, 'composite_friction.txt'), friction, fmt='%10.6f')
    np.savetxt(os.path.join(data_dir, 'friction_mapping.txt'), friction_mapping, fmt='%d')


    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                    friction, friction_mapping)

    # ===========  Solve for constraint forces  ============= #
    constraint_solver = ConstantSpeedConstraintSolver(param_file, SIM_STEP, max_step=SIM_STEP+2)
    
    # Particles to be optimized and their target angular velocity
    batch_speed = {i: 5 for i in range(sim.ngeom)}
    # batch_speed = {4:5}
    if not os.path.exists(os.path.join(data_dir, 'batch_speed.txt')):
        temp = np.stack([np.arange(sim.ngeom), np.zeros(sim.ngeom)]).T
    else:
        temp = np.loadtxt(os.path.join(data_dir, 'batch_speed.txt'))
    for k, v in batch_speed.items():
        temp[int(k)] = [k, v]
    np.savetxt(os.path.join(data_dir, 'batch_speed.txt'), temp, fmt='%6d  %10.5f')

    external_force = np.zeros((sim.ngeom, SIM_STEP, 2))
    external_torque = 0.8 * np.ones(sim.ngeom)
    
    if args.load_history:
        for i in batch_speed.keys():
            data = np.loadtxt(os.path.join(data_dir, "ft_%d.txt"%i))
            external_force[i] = data[:, 1:3]
            external_torque[i] = data[0, 3]

    constraint_solver.external_force.from_numpy(external_force)
    constraint_solver.external_torque.from_numpy(external_torque)

    # 1e-2 and 5e-8 works well
    optim_f = Momentum(external_force, lr=1e-2, bounds=[-15, 15], momentum=0.9, k=200, alpha=0.95)
    optim_t = Momentum(external_torque, lr=5e-8, bounds=[-2, 2], momentum=0.9, k=200, alpha=0.95)

    constraint_solver.input_parameters(hidden_state)
    constraint_solver.run(batch_speed, auto_diff=False, render=True)

    for i in range(NUM_ITER):
        constraint_solver.run(batch_speed, auto_diff=True)

        grad_f = constraint_solver.external_force.grad.to_numpy()
        grad_t = constraint_solver.external_torque.grad.to_numpy()
        external_force = optim_f.step(grad_f)
        external_torque = optim_t.step(grad_t)
        constraint_solver.external_force.from_numpy(external_force)
        constraint_solver.external_torque.from_numpy(external_torque)

        gpos = constraint_solver.geom_pos.to_numpy()[:, :SIM_STEP, :]
        rvel = constraint_solver.body_rvel.to_numpy()[:, :SIM_STEP]
        loss_gpos, loss_rvel = [], []
        for b in batch_speed.keys():
            loss_gpos.append(constraint_solver.loss_norm_factor[None] * \
                             np.sum(np.linalg.norm(gpos[b,1:SIM_STEP,b] - gpos[b,:SIM_STEP-1,b], axis=1)))
            loss_rvel.append(constraint_solver.loss_norm_factor[None] * \
                             constraint_solver.torque_norm_factor[None] * \
                             np.sum(np.abs(rvel[b] - rvel[b,0])))
        loss_gpos, loss_rvel = np.array(loss_gpos), np.array(loss_rvel)
        print("Iteration: %d, Loss: %.10f"%(i, constraint_solver.loss[None]), 
              ", loss_gpos: ", np.array_str(loss_gpos, precision=4),
              ", loss_rvel: ", np.array_str(loss_rvel, precision=4))
        if (i+1)%200 == 0:
            constraint_solver.run(batch_speed, auto_diff=False, render=True)
            for b in batch_speed.keys():
                gpos_error = np.average(np.linalg.norm(gpos[b,:,b] - gpos[b,0,b], axis=1))
                rvel_error = np.average(rvel[b] - rvel[b,0])
                print("Batch %d "%b, "gpos error: %.5f"%gpos_error,
                         "  rvel error: %.5f"%rvel_error)

        # Logging
        if (i+1) % args.save_every == 0:
            t = sim.dt * np.arange(SIM_STEP)
            for i in batch_speed.keys():
                ft_data = np.stack([t, external_force[i, :, 0], external_force[i, :, 1], 
                                    external_torque[i]*np.ones(SIM_STEP)]).T
                np.savetxt(os.path.join(data_dir, 'ft_%d.txt'%i), ft_data, fmt='%12.6f')