'''
Composite object pushing example
Interactions considered include spring force, damping force and friction force
'''
import os
import sys
import numpy as np
import taichi as ti
from .composite_util import Composite2D

@ti.data_oriented
class PushingSimulator:
    def __init__(self, composite, dt=1e-3):  # Initializer of the pushing environment
        self.dt = dt
        self.max_step = 512

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = -30, 30, -30, 30

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI(composite.obj_name, (self.resol_x, self.resol_y))

        self.ngeom = composite.num_particle + 1
        self.body_id2name = {0: composite.obj_name, 1: 'hand'}
        self.nbody = len(self.body_id2name.keys())

        # geom_body_id
        temp = np.concatenate([np.zeros(composite.num_particle), np.array([1])])
        self.geom_body_id = ti.field(ti.i32, shape=self.ngeom)
        self.geom_body_id.from_numpy(temp)

        # body_geom_id
        self.composite_geom_id = ti.field(ti.i32, shape=composite.num_particle)
        self.composite_geom_id.from_numpy(np.arange(composite.num_particle))
        self.hand_geom_id = ti.field(ti.i32, shape=())
        self.hand_geom_id = composite.num_particle
        
        self.mass = ti.field(ti.f64, composite.mass_dim)

        # hand mass
        self.hand_mass = 1e4
        self.hand_inertia = 1e4
        
        # contact parameters
        self.ks = 1e4
        self.eta = 10
        self.mu_s = 0.1
        self.mu_b = 0.3

        # pos, vel and force of each particle
        self.geom_pos = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.ngeom))
        self.geom_vel = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.ngeom))
        self.geom_force = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.ngeom))
        self.geom_pos0 = ti.Vector.field(2, ti.f64, shape=self.ngeom)
 
        self.body_qpos = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.nbody))
        self.body_qvel = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.nbody))
        self.body_rpos = ti.field(ti.f64, shape=(self.max_step, self.nbody))
        self.body_rvel = ti.field(ti.f64, shape=(self.max_step, self.nbody))

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, ti.f64, shape=(self.max_step, self.nbody))
        self.body_torque = ti.field(ti.f64, shape=(self.max_step, self.nbody))

        # radius of the particles
        self.radius = ti.field(ti.f64, self.ngeom)
        self.hand_radius = 2

        self.body_mass, self.body_inertia = ti.field(ti.f64, self.nbody), ti.field(ti.f64, self.nbody)

        self.composite = composite
        # composite particle pos0 in original polygon frame
        self.composite_p0 = ti.Vector.field(2, ti.f64, shape=composite.num_particle)
        for i in range(composite.num_particle):
            self.composite_p0[i] = composite.particle_pos0[i]

        # mapping from fine grid voxel to coarse voxel representation in composite object
        self.mapping = ti.field(ti.i32, shape=composite.num_particle)
        self.mapping.from_numpy(composite.mapping)

        # compute actions for composite in (0., 0.) 0. pose
        actions = composite.compute_actions(self.hand_radius)
        self.n_actions = actions['start_pos'].shape[0]
        self.pushing = ti.Vector.field(4, ti.f64, shape=self.n_actions)
        self.pushing.from_numpy(np.concatenate([actions['start_pos'], actions['direction']], axis=1))

        # hand initial velocity
        self.hand_vel = 20

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
    def collide(self, s: ti.i32):
        '''
        compute particle force based on pair-wise collision
        compute bottom friction force
        '''
        # clear force
        for i in range(self.ngeom):
            self.geom_force[s, i] = [0.0, 0.0]

        for i in range(self.ngeom):

            # bottom friction
            if self.geom_vel[s, i].norm() > 1e-5:
                # composite body
                if self.geom_body_id[i] == 0:
                    fb = - self.mu_b * self.mass[self.mapping[i]] * (self.geom_vel[s, i] / self.geom_vel[s, i].norm())
                    self.geom_force[s, i] += fb
                # hand
                elif self.geom_body_id[i] == 1:
                    fb = - 0 * self.hand_mass * (self.geom_vel[s, i] / self.geom_vel[s, i].norm())
                    self.geom_force[s, i] += fb

            pi = self.geom_pos[s, i]
            
            for j in range(self.ngeom):
                if self.geom_body_id[i] != self.geom_body_id[j]:
                    pj = self.geom_pos[s, j]
                    r_ij = pj - pi
                    r = r_ij.norm(1e-5)
                    n_ij = r_ij / r      # normal direction
                    if (self.radius[i] + self.radius[j] - r) > 0:
                        # spring force
                        fs = - self.ks * (self.radius[i] + self.radius[j] - r) * n_ij  
                        self.geom_force[s, i] += fs

                        # relative velocity
                        v_ij = self.geom_vel[s, j] - self.geom_vel[s, i]   
                        vn_ij = v_ij.dot(n_ij) * n_ij  # normal velocity
                        fd = self.eta * vn_ij   # damping force
                        self.geom_force[s, i] += fd

                        # side friction is activated with non-zero tangential velocity and non-breaking contact
                        vt_ij = v_ij - vn_ij   
                        if vn_ij.norm() > 1e-4 and v_ij.dot(n_ij) < -1e-4:
                            ft = self.mu_s * (fs.norm() + fd.norm()) * vt_ij / vn_ij.norm()
                            self.geom_force[s, i] += ft
                
    @ti.kernel
    def apply_external(self, s: ti.i32, geom_id: ti.i32, fx: ti.f64, fy: ti.f64):
        self.geom_force[s, geom_id][0] += fx
        self.geom_force[s, geom_id][1] += fy

    @ti.kernel
    def compute_ft(self, s: ti.i32):
        # compute the force torque on rigid bodies
        # clear net force and torque
        for i in range(self.nbody):
            self.body_force[s, i], self.body_torque[s, i] = [0.,0.], 0.

        for i in range(self.ngeom):
            body_id = self.geom_body_id[i]
            self.body_force[s, body_id] += self.geom_force[s, i]
            self.body_torque[s, body_id] += self.cross_2d(self.geom_force[s, i], 
                                                (self.body_qpos[s, body_id] - self.geom_pos[s, i]))

    @ti.kernel
    def update(self, s: ti.i32):
        for i in range(self.nbody):
            self.body_qvel[s+1, i] = self.body_qvel[s, i] + \
                                     self.dt * self.body_force[s, i] / self.body_mass[i]
            self.body_rvel[s+1, i] = self.body_rvel[s, i] + \
                                     self.dt * self.body_torque[s, i] / self.body_inertia[i]

            # update body qpos and rpos
            self.body_qpos[s+1, i] = self.body_qpos[s, i] + \
                                     self.dt * self.body_qvel[s+1, i] 
            self.body_rpos[s+1, i] = self.body_rpos[s, i] + \
                                     self.dt * self.body_rvel[s+1, i]

        # print(self.body_qpos[0, 0], self.body_qpos[0,1], self.body_qpos[1, 0], self.body_qpos[1,1],
        #      self.body_qpos[2, 0], self.body_qpos[2,1], '\n===================')

        for i in range(self.ngeom):
            body_id = self.geom_body_id[i]
            rot = self.rotation_matrix(self.body_rpos[s+1, body_id])
            self.geom_pos[s+1, i] = self.body_qpos[s+1, body_id] + rot @ self.geom_pos0[i]
            self.geom_vel[s+1, i] = self.body_qvel[s+1, body_id] + self.body_rvel[s+1, body_id] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    @ti.kernel
    def initialize(self, action_idx: ti.i32):
        self.place_composite()
        self.place_hand(action_idx)
        self.set_scene()

    @ti.func
    def place_composite(self):
        '''
        place the composite object in the world based on offset and rotation
        '''
        for i in self.composite_geom_id:
            self.geom_pos[0, i] = self.composite_p0[i]
            self.geom_vel[0, i] = [0., 0.]

    @ti.func
    def place_hand(self, action_idx):
        if action_idx < 0:
            action_idx = ti.cast(self.n_actions * ti.random(dtype=float), ti.i32)
        action = self.pushing[action_idx]

        ig = self.hand_geom_id
        self.geom_pos[0, ig][0] = action[0]  # x
        self.geom_pos[0, ig][1] = action[1]  # y
        self.geom_vel[0, ig][0] = self.hand_vel * action[2]  # vx
        self.geom_vel[0, ig][1] = self.hand_vel * action[3]  # vy

        ib = self.geom_body_id[ig]
        self.body_qpos[0, ib] = self.geom_pos[0, ig]
        self.body_qvel[0, ib] = self.geom_vel[0, ig]
        self.body_rpos[0, ib] = 0.
        self.body_rvel[0, ib] = 0.

    @ti.func
    def set_scene(self):
        ######################   composite body   #######################
        s, body_id, n = 0, 0, self.composite.num_particle
        for i in self.composite_geom_id:
            self.body_mass[body_id] += self.composite.fine_vsize**2 * self.mass[self.mapping[i]]

        # compute the body_qpos, body_qvel, body_rpos, body_rvel
        self.body_qpos[s, body_id] = [0., 0.]
        for i in self.composite_geom_id:
            self.body_qpos[s, body_id] += self.composite.fine_vsize**2 * self.mass[self.mapping[i]]\
                                         / self.body_mass[body_id] * self.geom_pos[s, i]
        self.body_qvel[s, body_id] = [0., 0.]
        self.body_rpos[s, body_id] = 0.
        self.body_rvel[s, body_id] = 0.

        # inertia
        self.body_inertia[body_id] = 0.
        for i in self.composite_geom_id:
            self.body_inertia[body_id] += self.composite.fine_vsize**2 * self.mass[self.mapping[i]] * \
                                0.01*(self.geom_pos[s, i] - self.body_qpos[s, body_id]).norm()**2

        for i in self.composite_geom_id:
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[s, i] - self.body_qpos[s, body_id]
            # radius
            self.radius[i] = self.composite.fine_vsize/2

        ######################   hand   #######################
        s, body_id = 0, 1
        self.body_mass[body_id] = self.hand_mass
        self.body_inertia[body_id] = self.hand_inertia
        self.radius[self.hand_geom_id] = self.hand_radius
        self.geom_pos0[self.hand_geom_id] = [0., 0.]        

    def render(self, s):  # Render the scene on GUI
        np_pos = self.geom_pos.to_numpy()[s]
        # print(np_pos[:10])
        np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                 (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

        # composite object
        r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
        idx = self.composite_geom_id.to_numpy()
        self.gui.circles(np_pos[idx], color=0xffffff, radius=r)

        # hand
        idx = self.hand_geom_id
        r = self.radius[idx] * self.resol_x / (self.wx_max-self.wx_min)
        self.gui.circles(np_pos[idx].reshape(1,2), color=0x09ffff, radius=r)

        self.gui.show()

