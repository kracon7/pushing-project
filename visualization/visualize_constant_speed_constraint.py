import os
import argparse
import cv2
import glob
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping
from taichi_pushing.physics.constant_speed_constraint_solver import ConstantSpeedConstraintSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults
import matplotlib.pyplot as plt

SIM_STEP = 50

ti.init(arch=ti.cpu, debug=True)
np.random.seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    parser.add_argument("--param_file", type=str, default='block_object_param.yaml')
    parser.add_argument("--data_dir", type=str, default='block_object_param_1')
    parser.add_argument("-i", type=int, default=0, help="particle index")
    args = parser.parse_args()

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    param_file = os.path.join(ROOT, 'config', args.param_file)
    sim = HiddenStateSimulator(param_file)

    data_dir = os.path.join(ROOT, 'data', args.data_dir)
    mass = np.loadtxt(os.path.join(data_dir, "composite_mass.txt"))
    friction = np.loadtxt(os.path.join(data_dir, "composite_friction.txt"))
    mass_mapping = np.loadtxt(os.path.join(data_dir, "mass_mapping.txt")).astype("int")
    friction_mapping = np.loadtxt(os.path.join(data_dir, "friction_mapping.txt")).astype("int")
    assert mass_mapping.shape[0] == sim.ngeom

    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                    friction, friction_mapping)

    external_ft = np.loadtxt(os.path.join(data_dir, "ft_%d.txt"%args.i))
    batch_speed = np.loadtxt(os.path.join(data_dir, 'batch_speed.txt'))
    
    u = {args.i: external_ft[:, 1:]}
    sim.input_parameters(hidden_state, u, [])
    sim.reset()
    sim.apply_initial_speed({args.i: batch_speed[args.i, 1]})
    sim.initialize()
    
    img_dir = os.path.join(os.path.expanduser('~'), 'tmp', 'taichi_vis')
    os.makedirs(img_dir, exist_ok=True)
    for f in glob.glob(os.path.join(img_dir, '*.jpg')):
        os.remove(f)

    for k in sim.u:
        for s in range(SIM_STEP):
            sim.bottom_friction(k, s)
            sim.apply_external(k, s, sim.u[k][s][0], sim.u[k][s][1], sim.u[k][s][2])
            sim.compute_ft(k, s)
            sim.forward_body(k, s)
            sim.forward_geom(k, s+1)
            img_path = os.path.join(img_dir, '%5d.jpg'%s)
            sim.render(k, s, img_path)

    img_list = os.listdir(img_dir)
    img_list = [f for f in img_list if f.endswith('.jpg')]
    img_list.sort()
    frameSize = (sim.resol_x, sim.resol_y)
    output_path = os.path.join(data_dir, 'vis_%d.avi'%args.i)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, frameSize)

    for i in range(len(img_list)):
        filename = os.path.join(img_dir, img_list[i])
        img = cv2.imread(filename)
        img = cv2.resize(img, frameSize, interpolation=cv2.INTER_AREA)
        out.write(img)

    out.release()
