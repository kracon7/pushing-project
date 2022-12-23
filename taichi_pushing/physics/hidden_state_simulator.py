'''
Block object rotation by the gripper using grasp-n-rotate motion
Simulated using the hidden states: Total mass, moment of inertia, center of mass, si (mu m g)
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
class HiddenStateSimulator:
    def __init__(self, param_file, max_step=100, dt=Defaults.DT):  # Initializer of the simulator environment
        self.block_object = BlockObject(param_file)
        self.dt = dt
        self.max_step = max_step

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = 0.05, 0.65, -0.3, 0.3

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI(self.block_object.params['shape'], (self.resol_x, self.resol_y))

        self.ngeom = self.block_object.num_particle
        self.nbody = 1

        # gravity
        self.gravity = ti.field(DTYPE, shape=())
        self.gravity[None] = 9.8

        self.composite_geom_id = ti.field(ti.i64, shape=self.ngeom)
        self.composite_geom_id.from_numpy(np.arange(self.ngeom))

        # composite mass and mass mapping
        self.composite_si = ti.field(DTYPE, self.ngeom, needs_grad=True) 
        self.si_mapping = ti.field(ti.i64, self.ngeom)

        # mass, pos, vel and force of each particle
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

        # Hidden states
        self.body_mass = ti.field(DTYPE, shape=(), needs_grad=True)
        self.body_inertia = ti.field(DTYPE, shape=(), needs_grad=True)
        self.body_com = ti.Vector.field(2, DTYPE, shape=(), needs_grad=True)
        self.geom_si = ti.field(DTYPE, self.ngeom, needs_grad=True)

        # composite particle pos0 in original pose
        self.composite_p0 = ti.Vector.field(2, DTYPE, shape=self.ngeom)
        self.composite_p0.from_numpy(self.block_object.particle_coord)

        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)
        self.loss_backtrack = ti.field(ti.f64, shape=())

        self.loss_norm_factor = ti.field(ti.f64, shape=())
        self.loss_norm_factor[None] = 1
        self.si_mapping_norm_factor = np.ones(self.ngeom)

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
            self.geom_si[i] = 0.

        for b, s in ti.ndrange(self.ngeom, self.max_step):
            self.body_qpos[b, s] = [0., 0.]
            self.body_qvel[b, s] = [0., 0.]
            self.body_rpos[b, s] = 0.
            self.body_rvel[b, s] = 0.
            self.body_force[b, s] = [0., 0.]
            self.body_torque[b, s] = 0.

    @ti.kernel
    def bottom_friction(self, b: ti.i32, s: ti.i32):
        # compute bottom friction force
        for i in range(self.ngeom):
            if self.geom_vel[b, s, i].norm() > 1e-6:
                self.geom_force[b, s, i] += - self.geom_si[i] * \
                                (self.geom_vel[b, s, i] / self.geom_vel[b, s, i].norm())

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

    @ti.kernel
    def forward_geom(self, b: ti.i32, s: ti.i32):
        for i in range(self.ngeom):
            rot = self.rotation_matrix(self.body_rpos[b, s+1])
            self.geom_pos[b, s+1, i] = self.body_qpos[b, s+1] + rot @ self.geom_pos0[i]
            self.geom_vel[b, s+1, i] = self.body_qvel[b, s+1] + self.body_rvel[b, s+1] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    @ti.kernel
    def initialize(self):
        # set initial geom_pos
        for b, i in ti.ndrange(self.ngeom, self.ngeom):
            self.geom_pos[b, 0, i] = self.composite_p0[i]

        for i in self.composite_geom_id:
            self.geom_si[i] = self.composite_si[self.si_mapping[i]]

        # compute the body_qpos
        for b in self.composite_geom_id:
            self.body_qpos[b, 0] = self.body_com[None]
        
        for i in self.composite_geom_id:
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[0, 0, i] - self.body_qpos[0, 0]
            # radius
            self.radius[i] = self.block_object.voxel_size / 2

    @ti.kernel
    def add_loss(self, b: ti.i32, s: ti.i32, vx: ti.f64, vy: ti.f64, vw: ti.f64):
        self.loss[None] += self.loss_norm_factor[None] * \
                           (ti.abs(self.body_qvel[b, s][0] - vx) + \
                            ti.abs(self.body_qvel[b, s][1] - vy) + \
                            ti.abs(self.body_rvel[b, s] - vw))

    @ti.kernel
    def add_loss_backtrack(self, b: ti.i32, s: ti.i64, i: ti.i32, px: ti.f64, py: ti.f64):
        self.loss_backtrack[None] += self.loss_norm_factor[None] * \
                                     ((self.geom_pos[b, s, i][0] - px)**2 + \
                                      (self.geom_pos[b, s, i][1] - py)**2 )
         
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

    def input_parameters(self, body_mass, body_inertia, body_com, composite_si, 
                        si_mapping, u, loss_steps):
        '''
        Set up simulation environment including mass, friction and control input u
        Args:
            body_mass -- float
            body_inertia -- float
            body_com -- (2, ) ndarray
                body center of mass
            composite_si -- (ngeom, ) ndarray
            si_mapping -- (ngeom, ) ndarray, 
                mapping to si
            u -- dict, 
                Control input. {particle_idx: list of [fx, fy, fw]}
            loss_steps -- list,
                Time steps to accumulate loss
        '''
        self.body_mass[None] = body_mass
        self.body_inertia[None] = body_inertia
        self.body_com.from_numpy(body_com)
        self.composite_si.from_numpy(composite_si)
        self.si_mapping.from_numpy(si_mapping)
        mapping_sum = [np.sum(si_mapping==i) for i in range(si_mapping.shape[0])]
        self.si_mapping_norm_factor = [1/i if i!=0 else 0 for i in mapping_sum]
        self.u = u
        self.loss_steps = loss_steps

    # def update_parameter(self, param, param_name):
    #     if param_name == "mass":
    #         self.composite_mass.from_numpy(param)
    #     elif param_name == "friction":
    #         self.composite_friction.from_numpy(param)
    #     else:
    #         raise Exception("Unknown parameter type")

    # def get_parameter(self, param_name):
    #     if param_name == "mass":
    #         param = self.composite_mass.to_numpy()
    #     elif param_name == "friction":
    #         param = self.composite_friction.to_numpy()
    #     else:
    #         raise Exception("Unknown parameter type")
    #     return param

    def run(self, sim_steps, render=False):
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

                if render:
                    self.render(k, s)

    def compute_loss(self, body_qvel_gt, body_rvel_gt):
        self.loss_norm_factor[None] = 1 / ( len(self.u.keys()) * 
                                      len(self.loss_steps) * self.ngeom)
        for b in self.u.keys():
            for s in self.loss_steps:
                self.add_loss(b, s, body_qvel_gt[b, s, 0], body_qvel_gt[b, s, 1],
                                body_rvel_gt[b, s])

    # def compute_loss_backtrack(self, geom_pos_gt):
    #     self.loss_norm_factor[None] = 1 / ( len(self.u.keys()) * 
    #                                   len(self.loss_steps) * self.ngeom)
    #     for b in self.u.keys():
    #         for s in self.loss_steps:
    #             for i in range(self.ngeom):
    #                 self.add_loss_backtrack(b, s, i, 
    #                                         geom_pos_gt[b, s, i, 0], 
    #                                         geom_pos_gt[b, s, i, 0])

    def map_to_hidden_state(self, composite_mass, mass_mapping, composite_friction, 
                            friction_mapping):
        '''
        From particle mass/friction to hidden states
        '''
        geom_mass, geom_friction = np.zeros(self.ngeom), np.zeros(self.ngeom)
        for i in range(self.ngeom):
            geom_mass[i] = composite_mass[mass_mapping[i]]
            geom_friction[i] = composite_friction[friction_mapping[i]]

        geom_pos = self.block_object.particle_coord
        body_mass = np.sum(geom_mass)
        body_com = np.zeros(2)
        for i in range(self.ngeom):
            body_com += geom_pos[i] * geom_mass[i] / body_mass

        body_inertia = 0.
        for i in range(self.ngeom):
            body_inertia += geom_mass[i] * np.linalg.norm(geom_pos[i] - body_com)**2

        si = np.zeros(self.ngeom)
        for i in range(self.ngeom):
            si[i] = geom_mass[i] * geom_friction[i] * self.gravity[None]
        
        hidden_state = {"body_mass": body_mass,
                        "body_inertia": body_inertia,
                        "body_com": body_com,
                        "si": si}
        return hidden_state