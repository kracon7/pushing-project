'''
Block object rotation by the gripper using grasp-n-rotate motion
Interaction is the bottom friction force
'''

import os
import sys
import numpy as np
import taichi as ti
from .block_object_util import BlockObject
from .utils import Defaults

DTYPE = Defaults.DTYPE

@ti.data_oriented
class GraspNRotateSimulator:
    def __init__(self, param_file, dt=Defaults.DT):  # Initializer of the simulator environment
        self.block_object = BlockObject(param_file)
        self.dt = dt
        self.max_step = 512

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = 0.05, 0.65, -0.3, 0.3

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI(self.block_object.params['shape'], (self.resol_x, self.resol_y))

        self.ngeom = self.block_object.num_particle
        self.nbody = 1

        # composite mass and mass mapping
        self.composite_mass = ti.field(DTYPE, self.ngeom, needs_grad=True) 
        self.mass_mapping = ti.field(ti.i64, self.ngeom)
        self.mass_mapping.from_numpy(self.block_object.mass_mapping)

        self.composite_geom_id = ti.field(ti.i64, shape=self.ngeom)
        self.composite_geom_id.from_numpy(np.arange(self.ngeom))

        # friction coefficient
        self.composite_friction = ti.field(DTYPE, self.ngeom, needs_grad=True)
        self.friction_mapping = ti.field(ti.i64, self.ngeom)
        self.friction_mapping.from_numpy(self.block_object.friction_mapping)
        self.geom_friction = ti.field(DTYPE, self.ngeom, needs_grad=True)

        # mass, pos, vel and force of each particle
        self.geom_mass = ti.field(DTYPE, self.ngeom, needs_grad=True)
        self.geom_pos = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step, self.ngeom), needs_grad=True)
        self.geom_vel = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step, self.ngeom), needs_grad=True)
        self.geom_force = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step, self.ngeom), needs_grad=True)
        self.geom_torque = ti.field(DTYPE, shape=(self.ngeom, self.max_step, self.ngeom), needs_grad=True)
        self.geom_pos0 = ti.Vector.field(2, DTYPE, shape=self.ngeom, needs_grad=True)
 
        self.body_qpos = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)
        self.body_qvel = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)
        self.body_rpos = ti.field(DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)
        self.body_rvel = ti.field(DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)
        self.body_torque = ti.field(DTYPE, shape=(self.ngeom, self.max_step), needs_grad=True)

        # radius of the particles
        self.radius = ti.field(DTYPE, self.ngeom)

        self.body_mass = ti.field(DTYPE, shape=(), needs_grad=True)
        self.body_inertia = ti.field(DTYPE, shape=(), needs_grad=True)

        # composite particle pos0 in original pose
        self.composite_p0 = ti.Vector.field(2, DTYPE, shape=self.ngeom)
        self.composite_p0.from_numpy(self.block_object.particle_coord)
        # for i in range(self.ngeom):
        #     self.composite_p0[i].from_numpy(self.block_object.particle_coord[i])

        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)
        self.loss_backtrack = ti.field(ti.f64, shape=())

    @staticmethod
    @ti.func
    def rotation_matrix(r):
        return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])

    @staticmethod
    @ti.func
    def cross_2d(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    @staticmethod
    @ti.func
    def right_orthogonal(v):
        return ti.Vector([-v[1], v[0]])

    @ti.kernel
    def clear_all(self):
        # clear force
        for b, s, i in ti.ndrange(self.ngeom, self.max_step, self.ngeom):
            self.geom_pos[b, s, i] = [0., 0.]
            self.geom_vel[b, s, i] = [0., 0.]
            self.geom_force[b, s, i] = [0., 0.]
            self.geom_torque[b, s, i] = 0.

        for i in range(self.ngeom):
            self.geom_pos0[i] = [0., 0.]
            self.geom_mass[i] = 0.

        for b, s in ti.ndrange(self.ngeom, self.max_step):
            self.body_qpos[b, s] = [0., 0.]
            self.body_qvel[b, s] = [0., 0.]
            self.body_rpos[b, s] = 0.
            self.body_rvel[b, s] = 0.
            self.body_force[b, s] = [0., 0.]
            self.body_torque[b, s] = 0.

        for i in range(self.nbody):
            self.body_mass[None] = 0.
            self.body_inertia[None] = 0.

    @ti.kernel
    def bottom_friction(self, b: ti.i32, s: ti.i32):
        # compute bottom friction force
        for i in range(self.ngeom):
            if self.geom_vel[b, s, i].norm() > 1e-8:
                fb = - self.geom_friction[i] * self.geom_mass[i] * (self.geom_vel[b, s, i] / self.geom_vel[b, s, i].norm())
                self.geom_force[b, s, i] += fb

    @ti.kernel
    def apply_external(self, b: ti.i32, s: ti.i32, fx: ti.f64, fy: ti.f64, fw: ti.f64):
        self.geom_force[b, s, b][0] += fx
        self.geom_force[b, s, b][1] += fy
        self.geom_torque[b, s, b] += fw
                    
    @ti.kernel
    def compute_ft(self, b: ti.i32, s: ti.i32):
        # compute the force torque on rigid bodies
        for i in range(self.ngeom):
            self.body_force[b, s] += self.geom_force[b, s, i]
            self.body_torque[b, s] += self.cross_2d(self.geom_force[b, s, i], 
                                                 (self.body_qpos[b, s] - self.geom_pos[b, s, i]))
            self.body_torque[b, s] += self.geom_torque[b, s, i]

    @ti.kernel
    def forward_body(self, b: ti.i32, s: ti.i32):
        self.body_qvel[b, s+1] = self.body_qvel[b, s] + \
                                    self.dt * self.body_force[b, s] / self.body_mass[None]
        self.body_rvel[b, s+1] = self.body_rvel[b, s] + \
                                    self.dt * self.body_torque[b, s] / self.body_inertia[None]

        # update body qpos and rpos
        self.body_qpos[b, s+1] = self.body_qpos[b, s] + \
                                    self.dt * self.body_qvel[b, s] 
        self.body_rpos[b, s+1] = self.body_rpos[b, s] + \
                                    self.dt * self.body_rvel[b, s]

        # print(self.body_qpos[0, 0], self.body_qpos[0,1], self.body_qpos[1, 0], self.body_qpos[1,1],
        #      self.body_qpos[2, 0], self.body_qpos[2,1], '\n===================')

    @ti.kernel
    def forward_geom(self, b: ti.i32, s: ti.i32):
        for i in range(self.ngeom):
            rot = self.rotation_matrix(self.body_rpos[b, s+1])
            self.geom_pos[b, s+1, i] = self.body_qpos[b, s+1] + rot @ self.geom_pos0[i]
            self.geom_vel[b, s+1, i] = self.body_qvel[b, s+1] + self.body_rvel[b, s+1] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    @ti.kernel
    def initialize(self):
        # set geom_pos
        for b, i in ti.ndrange(self.ngeom, self.ngeom):
            self.geom_pos[b, 0, i] = self.composite_p0[i]

        for i in self.composite_geom_id:
            self.geom_mass[i] = self.composite_mass[self.mass_mapping[i]]
            self.geom_friction[i] = self.composite_friction[self.friction_mapping[i]]

        #compute body mass and center of mass
        for i in self.composite_geom_id:
            self.body_mass[None] += self.geom_mass[i]

        # compute the body_qpos
        for b, i in ti.ndrange(self.ngeom, self.ngeom):
            self.body_qpos[b, 0] += self.geom_mass[i] / self.body_mass[None] * self.geom_pos[b, 0, i]
        
        for i in self.composite_geom_id:
            # inertia
            self.body_inertia[None] += self.geom_mass[i] * (self.geom_pos[0, 0, i] - self.body_qpos[0, 0]).norm()**2
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[0, 0, i] - self.body_qpos[0, 0]
            # radius
            self.radius[i] = self.block_object.voxel_size / 2

    @ti.kernel
    def compute_loss(self, b: ti.i32, idx: ti.i32, px: ti.f64, py: ti.f64, pw: ti.f64):
        self.loss[None] += (self.body_qpos[b, idx][0] - px)**2 + \
                           (self.body_qpos[b, idx][1] - py)**2 + \
                           (self.body_rpos[b, idx] - pw)**2

    @ti.kernel
    def compute_loss_backtrack(self, b: ti.i32, idx: ti.i64, px: ti.f64, py: ti.f64, pw: ti.f64):
        self.loss_backtrack[None] += (self.body_qpos[b, idx][0] - px)**2 + \
                                     (self.body_qpos[b, idx][1] - py)**2 + \
                                     (self.body_rpos[b, idx] - pw)**2                 
         
    def render(self, b, s):  # Render the scene on GUI
        np_pos = self.geom_pos.to_numpy()[b, s]
        # print(np_pos[:10])
        np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                 (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

        # composite object
        r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
        idx = self.composite_geom_id.to_numpy()
        self.gui.circles(np_pos[idx], color=0xffffff, radius=r)

        self.gui.show()

    def input_parameters(self, mass, mass_mapping, friction, friction_mapping, u):
        '''
        Set up simulation environment including mass, friction and control input u
        Args:
            mass -- (ngeom, ) ndarray, 
                    ground truth mass vector, usually only the first few
                    are mapped to the composite_mass
            mass_mapping -- (ngeom, ) ndarray, 
                            mapping to composite mass
            friction -- (ngeom, ) ndarray, 
                        ground truth friction vector, similar to "mass"
            friction_mapping -- (ngeom, ) ndarray, 
                                similar to "mass_mapping"
            u -- dict, 
                 Control input. {particle_idx: list of [fx, fy, fw]}
        '''
        self.composite_mass.from_numpy(mass)
        self.mass_mapping.from_numpy(mass_mapping)
        self.composite_friction.from_numpy(friction)
        self.friction_mapping.from_numpy(friction_mapping)
        self.u = u

    def run(self, sim_steps):
        for k in self.u:
            if sim_steps > len(self.u[k]):
                raise Exception("Undefined control input on particle %d, sim steps too long"%k)
        
        self.clear_all()
        self.initialize()
        for k in self.u:
            for s in range(sim_steps):
                self.bottom_friction(k, s)
                self.apply_external(k, s, self.u[k][s][0], self.u[k][s][1], self.u[k][s][2])
                self.compute_ft(k, s)
                self.forward_body(k, s)
                self.forward_geom(k, s)

    def backtracking(self, mass, grad, sim_steps, sim_gt, u, loss_steps, 
                           alpha=0.1, beta=0.8, lr_0=0.01, lr_min=1e-4):
        '''
        Backtracking line search for step size
        Args:
            mass -- numpy array of the current mass
            grad -- gradient at the current mass
            sim_steps -- number of simulation steps
            sim_gt -- ground truth results
            u -- control input list of [particle index, fx, fy, fw]
            loss_steps -- list of the timesteps added to loss computation
            alpha -- backtracking hypermeter
            beta -- backtracking step update ratio
            lr_0 -- step size at the beginning of search
            lr_min -- minimum step size to terminate the search
        Return:
            lr -- optimal step size
        '''
        
        lr = lr_0              # Initial learning rate
        loss_prev = 1e9        # 

        while lr >= lr_min:
            mass_new = mass - lr * grad
            if (mass_new <= 1e-2).any():
                print("Negative mass after update, gradient step is too large!")
                lr *= beta
                continue

            # Compute f(x')
            self.composite_mass.from_numpy(mass_new)

            self.loss_backtrack[None] = 0
            self.clear_all()
            self.initialize()
            for s in range(sim_steps):
                self.bottom_friction(s)
                self.apply_external(s, u[s][0], u[s][1], u[s][2], u[s][3])
                self.compute_ft(s)
                self.forward_body(s)
                self.forward_geom(s)
            
            for idx in loss_steps:
                self.compute_loss_backtrack(idx, 
                                            sim_gt.body_qpos[idx][0],
                                            sim_gt.body_qpos[idx][1], 
                                            sim_gt.body_rpos[idx])

            if self.loss[None] > self.loss_backtrack[None] and \
                                    self.loss_backtrack[None] > loss_prev:
                break
            else:
                lr *= beta

            loss_prev = self.loss_backtrack[None]
            print("Backtracking search at learning rate %.7f \
                    loss: %.9f, loss_backtracking: %.9f"%\
                    (lr, self.loss[None], self.loss_backtrack[None]))
        
        return lr