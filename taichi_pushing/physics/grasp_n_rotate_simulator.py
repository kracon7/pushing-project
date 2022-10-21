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
        self.geom_mass = ti.field(DTYPE, self.ngeom, needs_grad=True)

        self.composite_geom_id = ti.field(ti.i64, shape=self.ngeom)
        self.composite_geom_id.from_numpy(np.arange(self.ngeom))

        # contact parameters
        self.ks = 1e4
        self.eta = 10
        self.mu_s = 0.1
        self.mu_b = 0.5

        # pos, vel and force of each particle
        self.geom_pos = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_vel = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_force = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_torque = ti.field(DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_pos0 = ti.Vector.field(2, DTYPE, shape=self.ngeom, needs_grad=True)
 
        self.body_qpos = ti.Vector.field(2, DTYPE, shape=self.max_step, needs_grad=True)
        self.body_qvel = ti.Vector.field(2, DTYPE, shape=self.max_step, needs_grad=True)
        self.body_rpos = ti.field(DTYPE, shape=self.max_step, needs_grad=True)
        self.body_rvel = ti.field(DTYPE, shape=self.max_step, needs_grad=True)

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, DTYPE, shape=self.max_step, needs_grad=True)
        self.body_torque = ti.field(DTYPE, shape=self.max_step, needs_grad=True)

        # radius of the particles
        self.radius = ti.field(DTYPE, self.ngeom)

        self.body_mass = ti.field(DTYPE, shape=(), needs_grad=True)
        self.body_inertia = ti.field(DTYPE, shape=(), needs_grad=True)

        # composite particle pos0 in original pose
        self.composite_p0 = ti.Vector.field(2, DTYPE, shape=self.ngeom)
        self.composite_p0.from_numpy(self.block_object.particle_coord)
        # for i in range(self.ngeom):
        #     self.composite_p0[i].from_numpy(self.block_object.particle_coord[i])

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
        for s, i in ti.ndrange(self.max_step, self.ngeom):
            self.geom_pos[s, i] = [0., 0.]
            self.geom_vel[s, i] = [0., 0.]
            self.geom_force[s, i] = [0., 0.]
            self.geom_torque[s, i] = 0.

        for i in range(self.ngeom):
            self.geom_pos0[i] = [0., 0.]
            self.geom_mass[i] = 0.

        for s in range(self.max_step):
            self.body_qpos[s] = [0., 0.]
            self.body_qvel[s] = [0., 0.]
            self.body_rpos[s] = 0.
            self.body_rvel[s] = 0.
            self.body_force[s] = [0., 0.]
            self.body_torque[s] = 0.

        for i in range(self.nbody):
            self.body_mass[None] = 0.
            self.body_inertia[None] = 0.

    # @ti.kernel
    # def clear_grad(self):
    #     for s, i in ti.ndrange(self.max_step, self.ngeom):
    #         self.geom_pos.grad[s, i] = [0., 0.]
    #         self.geom_vel.grad[s, i] = [0., 0.]
    #         self.geom_force.grad[s, i] = [0., 0.]

    #     for i in range(self.ngeom):
    #         self.geom_pos0.grad[i] = [0., 0.]
    #         self.geom_mass.grad[i] = 0.

    #     for s, i in ti.ndrange(self.max_step, self.nbody):
    #         self.body_qpos.grad[s, i] = [0., 0.]
    #         self.body_qvel.grad[s, i] = [0., 0.]
    #         self.body_rpos.grad[s, i] = 0.
    #         self.body_rvel.grad[s, i] = 0.
    #         self.body_force.grad[s, i] = [0., 0.]
    #         self.body_torque.grad[s, i] = 0.

    #     for i in range(self.nbody):
    #         self.body_mass.grad[i] = 0.
    #         self.body_inertia.grad[i] = 0.

    #     for i in range(self.num_particle):
    #         self.composite_mass.grad[i] = 0.

    @ti.kernel
    def bottom_friction(self, s: ti.i64):
        # compute bottom friction force
        for i in range(self.ngeom):
            if self.geom_vel[s, i].norm() > 1e-8:
                fb = - self.mu_b * self.geom_mass[i] * (self.geom_vel[s, i] / self.geom_vel[s, i].norm())
                self.geom_force[s, i] += fb

    @ti.kernel
    def apply_external(self, s: ti.i64, geom_id: ti.i64, fx: ti.f64, fy: ti.f64, fw: ti.f64):
        self.geom_force[s, geom_id][0] += fx
        self.geom_force[s, geom_id][1] += fy
        self.geom_torque[s, geom_id] += fw
                    
    @ti.kernel
    def compute_ft(self, s: ti.i64):
        # compute the force torque on rigid bodies
        for i in range(self.ngeom):
            self.body_force[s] += self.geom_force[s, i]
            self.body_torque[s] += self.cross_2d(self.geom_force[s, i], 
                                                (self.body_qpos[s] - self.geom_pos[s, i]))
            self.body_torque[s] += self.geom_torque[s, i]

    @ti.kernel
    def forward_body(self, s: ti.i64):
        self.body_qvel[s+1] = self.body_qvel[s] + \
                                    self.dt * self.body_force[s] / self.body_mass[None]
        self.body_rvel[s+1] = self.body_rvel[s] + \
                                    self.dt * self.body_torque[s] / self.body_inertia[None]

        # update body qpos and rpos
        self.body_qpos[s+1] = self.body_qpos[s] + \
                                    self.dt * self.body_qvel[s] 
        self.body_rpos[s+1] = self.body_rpos[s] + \
                                    self.dt * self.body_rvel[s]

        # print(self.body_qpos[0, 0], self.body_qpos[0,1], self.body_qpos[1, 0], self.body_qpos[1,1],
        #      self.body_qpos[2, 0], self.body_qpos[2,1], '\n===================')

    @ti.kernel
    def forward_geom(self, s: ti.i64):
        for i in range(self.ngeom):
            rot = self.rotation_matrix(self.body_rpos[s+1])
            self.geom_pos[s+1, i] = self.body_qpos[s+1] + rot @ self.geom_pos0[i]
            self.geom_vel[s+1, i] = self.body_qvel[s+1] + self.body_rvel[s+1] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    @ti.kernel
    def initialize(self):
        # set geom_pos
        for i in self.composite_geom_id:
            self.geom_pos[0, i] = self.composite_p0[i]
            self.geom_mass[i] = self.composite_mass[self.mass_mapping[i]]

        #compute body mass and center of mass
        for i in self.composite_geom_id:
            self.body_mass[None] += self.geom_mass[i]

        # compute the body_qpos, body_qvel, body_rpos, body_rvel
        for i in self.composite_geom_id:
            self.body_qpos[0] += self.geom_mass[i] / self.body_mass[None] * self.geom_pos[0, i]
        
        for i in self.composite_geom_id:
            # inertia
            self.body_inertia[None] += self.geom_mass[i] * (self.geom_pos[0, i] - self.body_qpos[0]).norm()**2
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[0, i] - self.body_qpos[0]
            # radius
            self.radius[i] = self.block_object.voxel_size / 2

        
    def render(self, s):  # Render the scene on GUI
        np_pos = self.geom_pos.to_numpy()[s]
        # print(np_pos[:10])
        np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                 (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

        # composite object
        r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
        idx = self.composite_geom_id.to_numpy()
        self.gui.circles(np_pos[idx], color=0xffffff, radius=r)

        self.gui.show()