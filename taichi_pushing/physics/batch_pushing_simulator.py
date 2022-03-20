'''
Composite object pushing with a circle
The simulator simulates multple actions in a batch
Interactions considered include spring force, damping force and friction force
'''
import os
import sys
import numpy as np
import taichi as ti
from .composite_util import Composite2D
from .utils import Defaults

DTYPE = Defaults.DTYPE

@ti.data_oriented
class PushingSimulator:
    def __init__(self, composite, bs=1, dt=Defaults.DT):  # Initializer of the pushing environment
        self.batch_size = bs
        self.dt = dt
        self.max_step = 512

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = -30, 30, -30, 30

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.guis = []
        for i in range(bs):
            self.guis.append(ti.GUI(composite.obj_name+'_%d'%i, (self.resol_x, self.resol_y)))
        # self.gui = ti.GUI(composite.obj_name, (self.resol_x, self.resol_y))

        self.num_particle = composite.num_particle
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
        
        # hand mass
        self.hand_mass = 1e4
        self.hand_inertia = 1e4

        # composite mass and mass mapping
        self.composite_mass = ti.field(DTYPE, composite.mass_dim, needs_grad=True)
        self.mass_mapping = ti.field(ti.i32, composite.mass_dim)
        self.geom_mass = ti.field(DTYPE, self.ngeom, needs_grad=True)

        # contact parameters
        self.ks = 1e4
        self.eta = 10
        self.mu_s = 0.1
        self.mu_b = 0.5

        # pos, vel and force of each particle
        self.geom_pos = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.ngeom), needs_grad=True)
        self.geom_vel = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.ngeom), needs_grad=True)
        self.geom_force = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.ngeom), needs_grad=True)
        self.geom_torque = ti.field(DTYPE, shape=(bs, self.max_step, self.ngeom), needs_grad=True)
        self.geom_pos0 = ti.Vector.field(2, DTYPE, shape=self.ngeom, needs_grad=True)
 
        self.body_qpos = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)
        self.body_qvel = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)
        self.body_rpos = ti.field(DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)
        self.body_rvel = ti.field(DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)
        self.body_torque = ti.field(DTYPE, shape=(bs, self.max_step, self.nbody), needs_grad=True)

        # radius of the particles
        self.radius = ti.field(DTYPE, self.ngeom)
        self.hand_radius = 2

        self.body_mass = ti.field(DTYPE, self.nbody, needs_grad=True)
        self.body_inertia = ti.field(DTYPE, self.nbody, needs_grad=True)

        self.composite = composite
        # composite particle pos0 in original polygon frame
        self.composite_p0 = ti.Vector.field(2, DTYPE, shape=composite.num_particle)
        for i in range(composite.num_particle):
            self.composite_p0[i] = composite.particle_pos0[i]

        # compute actions for composite in (0., 0.) 0. pose
        actions = composite.compute_actions(self.hand_radius)
        self.n_actions = actions['start_pos'].shape[0]
        self.pushing = ti.Vector.field(4, DTYPE, shape=self.n_actions, needs_grad=True)
        self.pushing.from_numpy(np.concatenate([actions['start_pos'], actions['direction']], axis=1))

        # hand initial velocity
        self.hand_vel = 10

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
        for b, s, i in ti.ndrange(self.batch_size, self.max_step, self.ngeom):
            self.geom_pos[b, s, i] = [0., 0.]
            self.geom_vel[b, s, i] = [0., 0.]
            self.geom_force[b, s, i] = [0., 0.]
            self.geom_torque[b, s, i] = 0.

        for i in range(self.ngeom):
            self.geom_pos0[i] = [0., 0.]
            self.geom_mass[i] = 0.

        for b, s, i in ti.ndrange(self.batch_size, self.max_step, self.nbody):
            self.body_qpos[b, s, i] = [0., 0.]
            self.body_qvel[b, s, i] = [0., 0.]
            self.body_rpos[b, s, i] = 0.
            self.body_rvel[b, s, i] = 0.
            self.body_force[b, s, i] = [0., 0.]
            self.body_torque[b, s, i] = 0.

        for i in range(self.nbody):
            self.body_mass[i] = 0.
            self.body_inertia[i] = 0.

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
    def collide(self, s: ti.i32):
        '''
        compute particle force based on pair-wise collision
        compute bottom friction force
        '''
        for b, i in ti.ndrange(self.batch_size, self.ngeom):

            # bottom friction
            if self.geom_vel[b, s, i].norm() > 1e-5:
                fb = - self.mu_b * self.geom_mass[i] * (self.geom_vel[b, s, i] / self.geom_vel[b, s, i].norm())
                self.geom_force[b, s, i] += fb

            for j in range(self.ngeom):
                if self.geom_body_id[i] != self.geom_body_id[j]:
                    pi, pj = self.geom_pos[b, s, i], self.geom_pos[b, s, j]
                    r_ij = pj - pi
                    r = r_ij.norm()
                    n_ij = r_ij / r      # normal direction
                    if (self.radius[i] + self.radius[j] - r) > 0:
                        # spring force
                        fs = - self.ks * (self.radius[i] + self.radius[j] - r) * n_ij  
                        self.geom_force[b, s, i] += fs

                        # relative velocity
                        v_ij = self.geom_vel[b, s, j] - self.geom_vel[b, s, i]   
                        vn_ij = v_ij.dot(n_ij) * n_ij  # normal velocity
                        fd = self.eta * vn_ij   # damping force
                        self.geom_force[b, s, i] += fd

                        # # side friction is activated with non-zero tangential velocity and non-breaking contact
                        # vt_ij = v_ij - vn_ij   
                        # if vn_ij.norm() > 1e-4 and v_ij.dot(n_ij) < -1e-4:
                        #     ft = self.mu_s * (fs.norm() + fd.norm()) * vt_ij / vn_ij.norm()
                        #     self.geom_force[s, i] += ft

    @ti.kernel
    def apply_external(self, s: ti.i32, geom_id: ti.i32, fx: ti.f64, fy: ti.f64, fw: ti.f64):
        self.geom_force[s, geom_id][0] += fx
        self.geom_force[s, geom_id][1] += fy
        self.geom_torque[s, geom_id] += fw
                    
    @ti.kernel
    def compute_ft(self, s: ti.i32):
        # compute the force torque on rigid bodies
        for b, i in ti.ndrange(self.batch_size, self.ngeom):
            body_id = self.geom_body_id[i]
            self.body_force[b, s, body_id] += self.geom_force[b, s, i]
            self.body_torque[b, s, body_id] += self.cross_2d(self.geom_force[b, s, i], 
                                                (self.body_qpos[b, s, body_id] - self.geom_pos[b, s, i]))
            self.body_torque[b, s, body_id] += self.geom_torque[b, s, i]

    @ti.kernel
    def update(self, s: ti.i32):
        for b, i in ti.ndrange(self.batch_size, self.nbody):
            self.body_qvel[b, s+1, i] = self.body_qvel[b, s, i] + \
                                     self.dt * self.body_force[b, s, i] / self.body_mass[i]
            self.body_rvel[b, s+1, i] = self.body_rvel[b, s, i] + \
                                     self.dt * self.body_torque[b, s, i] / self.body_inertia[i]

            # update body qpos and rpos
            self.body_qpos[b, s+1, i] = self.body_qpos[b, s, i] + \
                                     self.dt * self.body_qvel[b, s, i] 
            self.body_rpos[b, s+1, i] = self.body_rpos[b, s, i] + \
                                     self.dt * self.body_rvel[b, s, i]

        # print(self.body_qpos[0, 0], self.body_qpos[0,1], self.body_qpos[1, 0], self.body_qpos[1,1],
        #      self.body_qpos[2, 0], self.body_qpos[2,1], '\n===================')

        for b, i in ti.ndrange(self.batch_size, self.ngeom):
            body_id = self.geom_body_id[i]
            rot = self.rotation_matrix(self.body_rpos[b, s+1, body_id])
            self.geom_pos[b, s+1, i] = self.body_qpos[b, s+1, body_id] + rot @ self.geom_pos0[i]
            self.geom_vel[b, s+1, i] = self.body_qvel[b, s+1, body_id] + self.body_rvel[b, s+1, body_id] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    def initialize(self, action_idx):
        '''
        Initialize simulator with a list of action index
        '''
        assert len(action_idx) == self.batch_size
        self.place_composite()
        for b, idx in enumerate(action_idx):
            self.place_hand(b, idx)


    @ti.kernel
    def place_hand(self, b: ti.i32, action_idx: ti.i32):
        ##################  Random Is Not Supported by AutoDiff Now  !!!!    ###################
        # if action_idx < 0:
        #     action_idx = ti.cast(self.n_actions * ti.random(dtype=float), ti.i32)
        ########################################################################################

        action = self.pushing[action_idx]

        ig = self.hand_geom_id
        ib = self.geom_body_id[ig]
        self.geom_pos[b, 0, ig][0] = action[0]  # x
        self.geom_pos[b, 0, ig][1] = action[1]  # y
        self.geom_vel[b, 0, ig][0] = self.hand_vel * action[2]  # vx
        self.geom_vel[b, 0, ig][1] = self.hand_vel * action[3]  # vy

        self.body_qpos[b, 0, ib] = self.geom_pos[b, 0, ig]
        self.body_qvel[b, 0, ib] = self.geom_vel[b, 0, ig]
        self.body_rpos[b, 0, ib] = 0.
        self.body_rvel[b, 0, ib] = 0.

        if b == 0:
            self.geom_mass[ig] = self.hand_mass
            self.body_mass[ib] = self.hand_mass
            self.body_inertia[ib] = self.hand_inertia
            self.radius[self.hand_geom_id] = self.hand_radius
            self.geom_pos0[self.hand_geom_id] = [0., 0.]        


    @ti.kernel
    def place_composite(self):
        # set geom_pos
        for b, i in ti.ndrange(self.batch_size, self.num_particle):
            # for i in self.composite_geom_id:
            self.geom_pos[b, 0, i] = self.composite_p0[i]

        for i in self.composite_geom_id:
            self.geom_mass[i] = self.composite_mass[self.mass_mapping[i]]

        #compute body mass and center of mass
        for i in self.composite_geom_id:
            self.body_mass[0] += self.composite.vsize**2 * self.geom_mass[i]

        # compute the body_qpos, body_qvel, body_rpos, body_rvel
        for b, i in ti.ndrange(self.batch_size, self.num_particle):
            # for i in self.composite_geom_id:
            self.body_qpos[b, 0, 0] += self.composite.vsize**2 * self.geom_mass[i]\
                                         / self.body_mass[0] * self.geom_pos[b, 0, i]
        
        for i in self.composite_geom_id:
            # inertia
            self.body_inertia[0] += self.composite.vsize**2 * self.geom_mass[i] * \
                                (self.geom_pos[0, 0, i] - self.body_qpos[0, 0, 0]).norm()**2
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[0, 0, i] - self.body_qpos[0, 0, 0]
            # radius
            self.radius[i] = self.composite.vsize/2

        
    def render(self, s):  # Render the scene on GUI
        batch_pos = self.geom_pos.to_numpy()[:, s, :, :]

        for b in range(self.batch_size):
            np_pos = batch_pos[b]
            np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                     (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

            # composite object
            r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
            idx = self.composite_geom_id.to_numpy()
            self.guis[b].circles(np_pos[idx], color=0xffffff, radius=r)

            # hand
            idx = self.hand_geom_id
            r = self.radius[idx] * self.resol_x / (self.wx_max-self.wx_min)
            self.guis[b].circles(np_pos[idx].reshape(1,2), color=0x09ffff, radius=r)

            self.guis[b].show()