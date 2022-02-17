import os
import sys
import numpy as np
import taichi as ti

ti.init()

DTYPE = ti.f32

@ti.data_oriented
class RigidBody:
    def __init__(self, dt=1e-3, num_particle=5, m=1):  # Initializer of the pushing environment
        self.dt = dt
        self.max_step = 512

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = -30, 30, -30, 30

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI('Rigid Body Collision', (self.resol_x, self.resol_y))

        self.num_particle = num_particle
        self.ngeom = num_particle + 1
        self.nbody = 2

        # geom_body_id
        temp = np.concatenate([np.zeros(num_particle), np.array([1])])
        self.geom_body_id = ti.field(ti.i32, shape=self.ngeom)
        self.geom_body_id.from_numpy(temp)

        # body_geom_id
        self.composite_geom_id = ti.field(ti.i32, shape=num_particle)
        self.composite_geom_id.from_numpy(np.arange(num_particle))
        self.hand_geom_id = ti.field(ti.i32, shape=())
        self.hand_geom_id = num_particle
        
        # mass of each circle
        self.mass = ti.field(DTYPE, self.ngeom, needs_grad=True)
        self.mass.from_numpy(m * np.ones(self.ngeom))
        
        # contact parameters and bottom friction coefficient
        self.ks = 1e4
        self.eta = 10
        self.mu_s = 0.1
        self.mu_b = 0.3

        # pos, vel and force of each circle
        self.geom_pos = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_vel = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_force = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_pos0 = ti.Vector.field(2, DTYPE, shape=self.ngeom, needs_grad=True)
 
        self.body_qpos = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)
        self.body_qvel = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)
        self.body_rpos = ti.field(DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)
        self.body_rvel = ti.field(DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)
        self.body_torque = ti.field(DTYPE, shape=(self.max_step, self.nbody), needs_grad=True)

        # radius of the particles
        temp = np.concatenate([np.ones(num_particle), np.array([3])])
        self.radius = ti.field(DTYPE, self.ngeom)
        self.radius.from_numpy(temp)

        self.body_mass = ti.field(DTYPE, self.nbody, needs_grad=True)
        self.body_inertia = ti.field(DTYPE, self.nbody, needs_grad=True)

        # initial velocity of the big circle
        self.init_vel = 10

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
            if self.geom_vel[s, i].norm(1e-5) > 1e-5:
                fb = - self.mu_b * self.mass[i] * (self.geom_vel[s, i] / self.geom_vel[s, i].norm(1e-5))
                self.geom_force[s, i] += fb

            for j in range(self.ngeom):
                if self.geom_body_id[i] != self.geom_body_id[j]:
                    pi, pj = self.geom_pos[s, i], self.geom_pos[s, j]
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
                        if vn_ij.norm(1e-5) > 1e-4 and v_ij.dot(n_ij) < -1e-4:
                            ft = self.mu_s * (fs.norm(1e-5) + fd.norm(1e-5)) * vt_ij / vn_ij.norm(1e-5)
                            self.geom_force[s, i] += ft
                

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

    def initialize(self):
        self.clear_composite_body()
        self.place_composite()
        self.place_hand()
        

    @ti.kernel
    def clear_composite_body(self):
        s, ib = 0, 0
        self.body_mass[ib] = 0.
        self.body_inertia[ib] = 0.
        self.body_qpos[s, ib] = [0., 0.]
        self.body_qvel[s, ib] = [0., 0.]
        self.body_rpos[s, ib] = 0.
        self.body_rvel[s, ib] = 0.


    @ti.kernel
    def place_hand(self):
        ##################  Random Is Not Supported by AutoDiff Now  !!!!    ###################
        # if action_idx < 0:
        #     action_idx = ti.cast(self.n_actions * ti.random(dtype=float), ti.i32)
        ########################################################################################

        ig = self.num_particle
        self.geom_pos[0, ig][0] = -5  # x
        self.geom_pos[0, ig][1] = 0  # y
        self.geom_vel[0, ig][0] = self.init_vel  # vx
        self.geom_vel[0, ig][1] = 0  # vy

        ib = 1
        self.body_qpos[0, ib] = self.geom_pos[0, ig]
        self.body_qvel[0, ib] = self.geom_vel[0, ig]
        self.body_rpos[0, ib] = 0.
        self.body_rvel[0, ib] = 0.

        self.body_mass[ib] = self.mass[ig]
        self.body_inertia[ib] = self.mass[ig]
        self.geom_pos0[ig] = [0., 0.]        


    @ti.kernel
    def place_composite(self):

        #compute body mass and center of mass
        for i in range(self.num_particle):
            self.body_mass[0] += self.mass[i]

        # set geom_pos
        for i in range(self.num_particle):
            self.geom_pos[0, i][0] = 0.
            self.geom_pos[0, i][1] = 2 * self.radius[i] * i
            self.geom_vel[0, i] = [0., 0.]

        
        # compute the body_qpos, body_qvel, body_rpos, body_rvel
        for i in range(self.num_particle):
            self.body_qpos[0, 0] += self.radius[i]**2 * self.mass[i]\
                                         / self.body_mass[0] * self.geom_pos[0, i]
        
        for i in range(self.num_particle):
            # inertia
            self.body_inertia[0] += self.radius[i]**2 * self.mass[i] * \
                                (self.geom_pos[0, i] - self.body_qpos[0, 0]).norm(1e-5)**2
            # geom_pos0
            self.geom_pos0[i] = self.geom_pos[0, i] - self.body_qpos[0, 0]

        
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
        idx = self.num_particle
        r = self.radius[idx] * self.resol_x / (self.wx_max-self.wx_min)
        self.gui.circles(np_pos[idx].reshape(1,2), color=0x09ffff, radius=r)

        self.gui.show()


@ti.data_oriented
class Loss():
    """docstring for Loss"""
    def __init__(self, engines):
        self.loss = ti.field(DTYPE, shape=(), needs_grad=True)
        self.engines = engines

    @ti.kernel
    def clear_loss(self):
        self.loss[None] = 0

    @ti.kernel
    def compute_loss(self, t: ti.i32):
        for i in range(self.engines[0].num_particle):
            self.loss[None] += (self.engines[0].geom_pos[t, i][0] - self.engines[1].geom_pos[t, i][0])**2 + \
                               (self.engines[0].geom_pos[t, i][1] - self.engines[1].geom_pos[t, i][1])**2


sim1 = RigidBody(m=1)
sim2 = RigidBody(m=2)
loss = Loss((sim1, sim2))


def run_world(sim):
    sim.initialize()
    for s in range(sim.max_step-1):
        sim.collide(s)
        sim.compute_ft(s)
        sim.update(s)
        # sim.render(s)

def forward():
    run_world(sim1)
    run_world(sim2)
    loss.compute_loss(500)

with ti.Tape(loss.loss):
    forward()

# forward()

print(loss.loss[None])
print(sim1.mass.grad)
