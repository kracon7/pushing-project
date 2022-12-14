'''
Compute the reactive constraint force with the block object rotation 
by the gripper using grasp-n-rotate motion.
The interaction is the bottom friction force.
'''

import os
import sys
import numpy as np
import taichi as ti
from .block_object_util import BlockObject
from .utils import Defaults

DTYPE = Defaults.DTYPE

@ti.data_oriented
class ConstraintForceSolver:
    def __init__(self, param_file, sim_step, max_step=100, dt=Defaults.DT):  # Initializer of the simulator environment
        self.block_object = BlockObject(param_file)
        self.dt = dt
        self.sim_step = sim_step
        self.max_step = max_step

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = 0.05, 0.65, -0.3, 0.3

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI(self.block_object.params['shape'], (self.resol_x, self.resol_y))

        self.ngeom = self.block_object.num_particle
        self.nbody = 1

        # composite mass and mass mapping
        self.composite_mass = ti.field(DTYPE, self.ngeom) 
        self.mass_mapping = ti.field(ti.i64, self.ngeom)
        self.mass_mapping.from_numpy(self.block_object.mass_mapping)

        self.composite_geom_id = ti.field(ti.i64, shape=self.ngeom)
        self.composite_geom_id.from_numpy(np.arange(self.ngeom))

        # friction coefficient
        self.composite_friction = ti.field(DTYPE, self.ngeom)
        self.friction_mapping = ti.field(ti.i64, self.ngeom)
        self.friction_mapping.from_numpy(self.block_object.friction_mapping)
        self.geom_friction = ti.field(DTYPE, self.ngeom)

        # mass, pos, vel and force of each particle
        self.geom_mass = ti.field(DTYPE, self.ngeom)
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

        self.external_force = ti.Vector.field(2, DTYPE, shape=(self.ngeom, sim_step), needs_grad=True)
        self.external_torque = ti.field(DTYPE, shape=(self.ngeom, sim_step))

        # radius of the particles
        self.radius = ti.field(DTYPE, self.ngeom)

        self.body_mass = ti.field(DTYPE, shape=())
        self.body_inertia = ti.field(DTYPE, shape=())

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
            if self.geom_vel[b, s, i].norm() > 1e-6:
                fb = - self.geom_friction[i] * self.geom_mass[i] * (self.geom_vel[b, s, i] / self.geom_vel[b, s, i].norm())
                self.geom_force[b, s, i] += fb

    @ti.kernel
    def apply_external(self, b: ti.i32, s: ti.i32):
        self.geom_force[b, s, b][0] += self.external_force[b, s][0]
        self.geom_force[b, s, b][1] += self.external_force[b, s][1]
        self.geom_torque[b, s, b] += self.external_torque[b, s]
                    
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
    def add_loss(self, b: ti.i32, s: ti.i32):
        self.loss[None] += (self.geom_pos[b, s, b][0] - self.geom_pos[b, s+1, b][0])**2 + \
                            (self.geom_pos[b, s, b][1] - self.geom_pos[b, s+1, b][1])**2

    @ti.kernel
    def average_loss(self, n: ti.f64):
        self.loss[None] /= n
         
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

    def input_parameters(self, mass, mass_mapping, friction, friction_mapping, 
                                external_force, external_torque):
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
            external_force -- (ngeom, max_steps, 2) ndarray
            external_torque -- (ngeom, max_steps, ) ndarray
        '''
        self.composite_mass.from_numpy(mass)
        self.mass_mapping.from_numpy(mass_mapping)
        self.composite_friction.from_numpy(friction)
        self.friction_mapping.from_numpy(friction_mapping)
        self.external_force.from_numpy(external_force)
        self.external_torque.from_numpy(external_torque)

    def update_parameter(self, param, param_name):
        if param_name == "mass":
            self.composite_mass.from_numpy(param)
        elif param_name == "friction":
            self.composite_friction.from_numpy(param)
        elif param_name == "force":
            self.external_force.from_numpy(param)
        else:
            raise Exception("Unknown parameter type")

    def get_parameter(self, param_name):
        if param_name == "mass":
            param = self.composite_mass.to_numpy()
        elif param_name == "friction":
            param = self.composite_friction.to_numpy()
        elif param_name == "force":
            param = self.external_force.to_numpy()
        else:
            raise Exception("Unknown parameter type")
        return param

    def run(self, batch, render=False):
        self.clear_all()
        self.initialize()
        for b in batch:
            for s in range(self.sim_step):
                self.bottom_friction(b, s)
                self.apply_external(b, s)
                self.compute_ft(b, s)
                self.forward_body(b, s)
                self.forward_geom(b, s)

                if render:
                    self.render(b, s)

    def compute_loss(self, batch):
        n = 0
        for b in batch:
            for s in range(self.sim_step):
                self.add_loss(b, s)
                n += 1

        self.average_loss(n)