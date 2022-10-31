import os
import sys
import argparse
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.grasp_n_rotate_simulator import GraspNRotateSimulator

ti.init(arch=ti.cpu, debug=True)

def time_alignment(ft, pose):    
    ft_time = ft[:, 0]
    pose_time = pose[:, 0]

    fx_interp = np.interp(pose_time, ft_time, ft[:, 1])
    fy_interp = np.interp(pose_time, ft_time, ft[:, 2])
    fz_interp = np.interp(pose_time, ft_time, ft[:, 3])
    tx_interp = np.interp(pose_time, ft_time, ft[:, 4])
    ty_interp = np.interp(pose_time, ft_time, ft[:, 5])
    tz_interp = np.interp(pose_time, ft_time, ft[:, 6])

    force_interp = np.stack([fx_interp, fy_interp, fz_interp]).T
    torque_interp = np.stack([tx_interp, ty_interp, tz_interp]).T
    return force_interp, torque_interp

def convert_rotation(pose):
    '''
    Compute theta_z in euler angle and 2D rotation matrix from pose quaternion
    '''
    r_ez, rmat = [], []
    for p in pose:
        euler = Rotation.from_quat(p[4:]).as_euler('zyx')
        theta_z = euler[0] + np.pi / 2
        cosz, sinz = np.cos(theta_z), np.sin(theta_z)
        r = np.array([[cosz, -sinz],
                        [sinz, cosz]])
        r_ez.append(theta_z)
        rmat.append(r)
    r_ez = np.stack(r_ez)
    rmat = np.stack(rmat)
    return r_ez, rmat

def rotate_force(force, rmat):
    '''
    Rotate the force into robot base frame
    '''
    rotated_fx, rotated_fy = [], []
    for f, r in zip(force, rmat):
        rotated_fx_fy = r.T @ np.array([f[0], f[1]])
        rotated_fx.append(rotated_fx_fy[0])
        rotated_fy.append(rotated_fx_fy[1])

    rotated_fx = np.stack(rotated_fx)
    rotated_fy = np.stack(rotated_fy)
    return rotated_fx, rotated_fy


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot real robot force/torque and pose data')
    parser.add_argument('--data', type=str, help='dir for force torque file and pose file')
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)
    
    mass = 0.1 * np.ones(sim.block_object.num_particle)
    # mass[:4] = np.array([1, 1.5, 2, 2.5]) 
    sim.composite_mass.from_numpy(mass)

    # # map mass to THREE regions 
    mapping = np.zeros(sim.ngeom)
    # mapping[int(sim.num_particle/4):] += 1
    # mapping[int(2*sim.num_particle/4):] += 1
    # mapping[int(3*sim.num_particle/4):] += 1

    sim.mass_mapping.from_numpy(mapping)
    
    # Load force torque data
    ft = np.loadtxt(os.path.join(args.data, 'ft300s.txt'), delimiter=',')
    pose = np.loadtxt(os.path.join(args.data, 'pose.txt'), delimiter=',')
    force_interp, torque_interp = time_alignment(ft, pose)
    r_ez, rmat = convert_rotation(pose)
    rotated_fx, rotated_fy = rotate_force(force_interp, rmat)
    rot_origin = pose[0, 1:3]

    particle_origin_idx = int(np.argmin(np.linalg.norm(
                                        sim.block_object.particle_coord - rot_origin, 
                                        axis=1)))

    # Test forward simulation
    sim.clear_all()
    sim.initialize()
    for s in range(500):
        sim.bottom_friction(s)
        sim.apply_external(s, particle_origin_idx, rotated_fx[s], rotated_fy[s], torque_interp[s, 2])
        sim.compute_ft(s)
        sim.forward_body(s)
        sim.forward_geom(s)
        sim.render(s)

    # # Test auto-differentiation
    # sim.clear_all()
    # loss = ti.field(ti.f64, shape=(), needs_grad=True)
    # loss[None] = 0

    # @ti.kernel
    # def compute_loss():
    #     loss[None] = sim.body_qpos[0].norm()**2 + sim.body_rpos[0]**2

    # with ti.ad.Tape(loss):
    #     sim.initialize()
    #     for s in range(20):
    #         sim.bottom_friction(s)
    #         sim.apply_external(s, 10, 0, 0, 100)
    #         sim.compute_ft(s)
    #         sim.forward_body(s)
    #         sim.forward_geom(s)

    #     compute_loss()

    # print('loss: ', loss[None], 
    #       " dl/dqx: ", sim.body_qpos.grad.to_numpy()[0],
    #       " dl/drx: ", sim.body_rpos.grad.to_numpy()[0],
    #       " at qpos: ", sim.body_qpos.to_numpy()[0],
    #       " rpos: ", sim.body_rpos.to_numpy()[0])