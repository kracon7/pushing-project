import os
import sys
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot real robot force/torque and pose data')
    parser.add_argument('--data', type=str, help='dir for force torque file and pose file')
    parser.add_argument('--block', type=str, help="block object config file")
    args = parser.parse_args()

    ft = np.loadtxt(os.path.join(args.data, 'ft300s.txt'), delimiter=',')
    pose = np.loadtxt(os.path.join(args.data, 'pose.txt'), delimiter=',')
    force_interp, torque_interp = time_alignment(ft, pose)
    r_ez, rmat = convert_rotation(pose)
    rotated_fx, rotated_fy = rotate_force(force_interp, rmat)
    rot_origin = pose[0, 1:3]

    # Use relative time for plot
    pose_time = pose[:, 0]
    relative_time = pose_time - pose_time[0]

    nstep = pose.shape[0]
    
    fig = make_subplots(
        rows=4, cols=1, subplot_titles=('Force Feedback', 'Torque Feedback', 'Object Particles in Robot Frame'),
        horizontal_spacing=0.02, vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    fig.update_xaxes(title="time (s)", row=1, col=1)
    fig.update_yaxes(title="force (N)", row=1, col=1)
    fig.update_xaxes(title="time (s)", row=2, col=1)
    fig.update_yaxes(title="torque (N*m)", row=2, col=1)
    fig.update_xaxes(range=[0.05, 0.85], title="X (m)", row=3, col=1)
    fig.update_yaxes(range=[-0.4, 0.4], title="Y (m)", row=3, col=1)
    fig.add_trace(go.Scatter(x=relative_time, y=rotated_fx, name='fx'), row=1, col=1)
    fig.add_trace(go.Scatter(x=relative_time, y=rotated_fy, name='fy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=relative_time, y=torque_interp[:,2], name='tz'), row=2, col=1)

    fig.update_layout(title="Grasp and Slide Action", 
                      hovermode="closest",
                      width = 500,
                      height = 1600
                     )

    
    fig.show()