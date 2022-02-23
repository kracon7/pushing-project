# single unknown for the mass

import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.utils import Defaults

np.random.seed(0)

DTYPE = Defaults.DTYPE

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class ParticleSimulator:
    def __init__(self, layers, dt=Defaults.DT):  # Initializer of the pushing environment
        self.dt = dt
        self.max_step = 512

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = -30, 30, -30, 30

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI('billards', (self.resol_x, self.resol_y))

        self.layers = layers
        self.ngeom = int(layers * (layers+1) / 2) + 1
        self.nbody = self.ngeom

        # geom_body_id
        temp = np.arange(self.ngeom)
        self.geom_body_id = ti.field(ti.i32, shape=self.ngeom)
        self.geom_body_id.from_numpy(temp)

        self.geom_mass = ti.field(DTYPE, self.ngeom, needs_grad=True)

        # contact parameters
        self.ks = 1e4
        self.eta = 10
        self.mu_s = 0.1
        self.mu_b = 0.3

        # pos, vel and force of each particle
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
        self.radius = ti.field(DTYPE, self.ngeom)

        self.body_mass = ti.field(DTYPE, self.nbody, needs_grad=True)
        self.body_inertia = ti.field(DTYPE, self.nbody, needs_grad=True)

        # hand initial velocity
        self.hand_vel = 100

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

        for i in range(self.ngeom):
            self.geom_pos0[i] = [0., 0.]

        for s, i in ti.ndrange(self.max_step, self.nbody):
            self.body_qpos[s, i] = [0., 0.]
            self.body_qvel[s, i] = [0., 0.]
            self.body_rpos[s, i] = 0.
            self.body_rvel[s, i] = 0.
            self.body_force[s, i] = [0., 0.]
            self.body_torque[s, i] = 0.

        for i in range(self.nbody):
            self.body_mass[i] = 0.
            self.body_inertia[i] = 0.


    @ti.kernel
    def collide(self, s: ti.i32):
        '''
        compute particle force based on pair-wise collision
        compute bottom friction force
        '''
        for i in range(self.ngeom):
            # bottom friction
            if self.geom_vel[s, i].norm() > 1e-5:
                fb = - self.mu_b * self.geom_mass[i] * (self.geom_vel[s, i] / self.geom_vel[s, i].norm())
                self.geom_force[s, i] += fb

            for j in range(self.ngeom):
                if self.geom_body_id[i] != self.geom_body_id[j]:
                    pi, pj = self.geom_pos[s, i], self.geom_pos[s, j]
                    r_ij = pj - pi
                    r = r_ij.norm()
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

                        # # side friction is activated with non-zero tangential velocity and non-breaking contact
                        # vt_ij = v_ij - vn_ij   
                        # if vn_ij.norm() > 1e-4 and v_ij.dot(n_ij) < -1e-4:
                        #     ft = self.mu_s * (fs.norm() + fd.norm()) * vt_ij / vn_ij.norm()
                        #     self.geom_force[s, i] += ft
                    
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
                                     self.dt * self.body_qvel[s, i] 
            self.body_rpos[s+1, i] = self.body_rpos[s, i] + \
                                     self.dt * self.body_rvel[s, i]

        # print(self.body_qpos[0, 0], self.body_qpos[0,1], self.body_qpos[1, 0], self.body_qpos[1,1],
        #      self.body_qpos[2, 0], self.body_qpos[2,1], '\n===================')

        for i in range(self.ngeom):
            body_id = self.geom_body_id[i]
            rot = self.rotation_matrix(self.body_rpos[s+1, body_id])
            self.geom_pos[s+1, i] = self.body_qpos[s+1, body_id] + rot @ self.geom_pos0[i]
            self.geom_vel[s+1, i] = self.body_qvel[s+1, body_id] + self.body_rvel[s+1, body_id] \
                                            * self.right_orthogonal(rot @ self.geom_pos0[i])

    def initialize(self):
        self.clear_all()
        self.set_scene(2)

    @ti.kernel
    def set_scene(self, r: ti.f64):
        count = 0
        for i in range(self.layers):
            for j in range(i + 1):
                count += 1
                self.geom_pos[0, count] = [
                    5 + i * 2 * r, j * 2 * r - i * r * 0.7
                ]

        self.geom_pos[0, 0] = [0, 0]
        self.geom_vel[0, 0][0] = self.hand_vel  # vx
        self.geom_vel[0, 0][1] = 0  # vy

    @ti.kernel
    def set_mass(self):
        # set body_pos and body_vel
        for i in range(self.ngeom):
            self.body_qpos[0, i] = self.geom_pos[0, i]
            self.body_qvel[0, i] = self.geom_vel[0, i]

            # set body_mass and body_inertia
            self.body_mass[i] = self.geom_mass[i]
            self.body_inertia[i] = self.geom_mass[i]

            # set radius and geom_pos0
            self.radius[i] = 2

        
    def render(self, s):  # Render the scene on GUI
        np_pos = self.geom_pos.to_numpy()[s]
        # print(np_pos[:10])
        np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                 (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

        # composite object
        r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
        idx = np.arange(1, self.ngeom)
        self.gui.circles(np_pos[idx], color=0xffffff, radius=r)

        # hand
        idx = 0
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
        for i in range(self.engines[0].ngeom):
            self.loss[None] += (self.engines[0].geom_pos[t, i][0] - self.engines[1].geom_pos[t, i][0])**2 + \
                    (self.engines[0].geom_pos[t, i][1] - self.engines[1].geom_pos[t, i][1])**2


def run_world(sim):
    sim.set_mass()
    for s in range(sim.max_step-1):
        sim.collide(s)
        sim.compute_ft(s)
        sim.update(s)
        # sim.render(s)

def run_episode():
    run_world(sim_est)
    run_world(sim_gt)

def forward():
    run_episode()
    loss.compute_loss(300)


sim_est, sim_gt = ParticleSimulator(2), ParticleSimulator(2)

mass_gt, mass_est = np.ones(sim_est.ngeom), np.ones(sim_est.ngeom)
mass_est[0] = 1.1918
# mass_est[0] = 1.0967
sim_gt.geom_mass.from_numpy(mass_gt)
sim_est.geom_mass.from_numpy(mass_est)

loss = Loss((sim_est, sim_gt))

lr = 5e-5
max_iter = 100

for i in range(max_iter):
    sim_gt.initialize()
    sim_est.initialize()
    loss.clear_loss()

    # forward sims to compute loss and gradient
    with ti.Tape(loss.loss):
        forward()

    print('Iteration %d loss: %12.4f estimated mass: %16.8f gradient: %.4f'%(
            i, loss.loss[None], sim_est.geom_mass[0], sim_est.geom_mass.grad[0]))

    grad = sim_est.geom_mass.grad

    # update estimated mass
    if np.abs(grad.to_numpy()[0]) < 1e4:
        mass_est[0] -= lr * grad.to_numpy()[0]
    else:
        mass_est[0] += 1e-5
    sim_est.geom_mass.from_numpy(mass_est)

# sim = ParticleSimulator(2)

# sim.geom_mass.from_numpy(np.ones(sim.ngeom))

# sim.initialize()

# for s in range(sim.max_step-1):
#     sim.collide(s)
#     sim.compute_ft(s)
#     sim.update(s)
#     sim.render(s)
