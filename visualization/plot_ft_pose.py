import os
import sys
import argparse
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.block_object_util import BlockObject

def rotate_block(coord, pose, origin):
    # rotate the particle coordinates
    euler = Rotation.from_quat(pose).as_euler('zyx')
    theta_z = euler[0] + np.pi / 2
    cosz, sinz = np.cos(theta_z), np.sin(theta_z)
    R = np.array([[cosz, -sinz],
                    [sinz, cosz]])
    rotated_coord = (coord - origin) @ R + rot_origin
    return rotated_coord

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot real robot force/torque and pose data')
    parser.add_argument('--ft', type=str, help='force torque file')
    parser.add_argument('--pose', type=str, help='pose file file')
    parser.add_argument('--block', type=str, help="block object config file")
    args = parser.parse_args()

    ft = np.loadtxt(args.ft, delimiter=',')
    pose = np.loadtxt(args.pose, delimiter=',')
    rot_origin = pose[0, 1:3]

    ft_time = ft[:, 0]
    pose_time = pose[:, 0]

    ft_interp_fx = np.interp(pose_time, ft_time, ft[:, 1])
    ft_interp_fy = np.interp(pose_time, ft_time, ft[:, 2])
    ft_interp_fz = np.interp(pose_time, ft_time, ft[:, 3])
    ft_interp_tx = np.interp(pose_time, ft_time, ft[:, 4])
    ft_interp_ty = np.interp(pose_time, ft_time, ft[:, 5])
    ft_interp_tz = np.interp(pose_time, ft_time, ft[:, 6])

    nstep = pose.shape[0]

    block_object = BlockObject(args.block)
    coord = block_object.particle_coord

    rotated_coord = rotate_block(coord, pose[0, 4:], rot_origin)
    
    fig = make_subplots(
        rows=3, cols=1, subplot_titles=('Force Feedback', 'Torque Feedback', 'Object Particles in Robot Frame'),
        horizontal_spacing=0.02, vertical_spacing=0.08,
        row_heights=[0.25, 0.25, 0.5]
    )
    fig.update_xaxes(title="time (s)", row=1, col=1)
    fig.update_yaxes(title="force (N)", row=1, col=1)
    fig.update_xaxes(title="time (s)", row=2, col=1)
    fig.update_yaxes(title="torque (N*m)", row=2, col=1)
    fig.update_xaxes(range=[-0.05, 0.55], title="X (m)", row=3, col=1)
    fig.update_yaxes(range=[-0.3, 0.3], title="Y (m)", row=3, col=1)
    fig.add_trace(go.Scatter(x=pose_time, y=ft_interp_fx, name='fx'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pose_time, y=ft_interp_fy, name='fy'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pose_time, y=ft_interp_tz, name='tz'), row=2, col=1)
    fig.add_trace(go.Scatter(x=rotated_coord[:, 0], y=rotated_coord[:, 1], mode = 'markers', 
                             name='block object', marker=dict(size=16)), 
                  row=3, col=1)

    frames = []
    for i in range(0, nstep, 10):
        # rotate the particle coordinates
        rotated_coord = rotate_block(coord, pose[i, 4:], rot_origin)
        # add frame
        frames.append(go.Frame(data=[go.Scatter(x=pose_time[:i], y=ft_interp_fx[:i]),
                                     go.Scatter(x=pose_time[:i], y=ft_interp_fy[:i]),
                                     go.Scatter(x=pose_time[:i], y=ft_interp_tz[:i]),
                                     go.Scatter(x=rotated_coord[:, 0], y=rotated_coord[:, 1], mode = 'markers')],
                                traces=[0,1,2,3]))
    fig.update(frames=frames)
    
    fig.update_layout(go.Layout(
                title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
                width = 900,
                height = 1600,
                updatemenus=[dict(type="buttons",
                                buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None])])]),)
    
    fig.show()