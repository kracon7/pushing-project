# two circles collision

import os
import sys
import numpy as np
import taichi as ti

DTYPE = ti.f64

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class ParticleSimulator:
    def __init__(self, dt=1e-3):
        self.dt = dt
        self.max_step = 200

        # world bound
        self.wx_min, self.wx_max, self.wy_min, self.wy_max = -30, 30, -30, 30

        # render resolution
        self.resol_x, self.resol_y = 800, 800
        self.gui = ti.GUI('billards', (self.resol_x, self.resol_y))

        self.ngeom = 2

        self.geom_mass = ti.field(DTYPE, self.ngeom, needs_grad=True)

        # contact parameters
        self.ks = 1e4     # spring constant for contact

        # pos, vel and force of each particle
        self.geom_pos = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_vel = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)
        self.geom_force = ti.Vector.field(2, DTYPE, shape=(self.max_step, self.ngeom), needs_grad=True)

        # radius of the particles
        self.radius = ti.field(DTYPE, self.ngeom)
        self.radius.from_numpy(2*np.ones(self.ngeom))   # circle radius is 2


    @ti.kernel
    def collide(self, s: ti.i32):
        '''
        compute particle force based on pair-wise collision
        '''
        for i in range(self.ngeom):
            for j in range(self.ngeom):
                if i != j:
                    pi, pj = self.geom_pos[s, i], self.geom_pos[s, j]
                    r_ij = pj - pi
                    r = r_ij.norm()
                    n_ij = r_ij / r      # normal direction
                    if (self.radius[i] + self.radius[j] - r) > 0:
                        # spring force
                        fs = - self.ks * (self.radius[i] + self.radius[j] - r) * n_ij  
                        self.geom_force[s, i] += fs


    @ti.kernel
    def update(self, s: ti.i32):
        # update the next step velocity and position
        for i in range(self.ngeom):
            self.geom_vel[s+1, i] = self.geom_vel[s, i] + \
                                     self.dt * self.geom_force[s, i] / self.geom_mass[i]
            self.geom_pos[s+1, i] = self.geom_pos[s, i] + \
                                     self.dt * self.geom_vel[s, i]
            
    def initialize(self):
        # clear states and set the simulation scene
        self.clear_all()
        self.set_scene()

    @ti.kernel
    def clear_all(self):
        # clear all position velocity and gradients
        for s, i in ti.ndrange(self.max_step, self.ngeom):
            self.geom_pos[s, i] = [0., 0.]
            self.geom_vel[s, i] = [0., 0.]
            self.geom_force[s, i] = [0., 0.]
            self.geom_pos.grad[s, i] = [0., 0.]
            self.geom_vel.grad[s, i] = [0., 0.]
            self.geom_force.grad[s, i] = [0., 0.]

        for i in range(self.ngeom):
            self.geom_mass.grad[i] = 0
            

    @ti.kernel
    def set_scene(self):
        # place two circles in the scene and add initial velocity to the first circle
        self.geom_pos[0, 0] = [0, 0]
        self.geom_pos[0, 1] = [4, 0]
        self.geom_vel[0, 0][0] = 100  # vx
        self.geom_vel[0, 0][1] = 0  # vy

        
    def render(self, s):  # Render the scene on GUI
        np_pos = self.geom_pos.to_numpy()[s]
        # print(np_pos[:10])
        np_pos = (np_pos - np.array([self.wx_min, self.wy_min])) / \
                 (np.array([self.wx_max-self.wx_min, self.wy_max-self.wy_min]))

        # composite object
        r = self.radius[0] * self.resol_x / (self.wx_max-self.wx_min)
        self.gui.circles(np_pos, color=0xffffff, radius=r)

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
    for s in range(sim.max_step-1):
        sim.collide(s)
        sim.update(s)

def forward():
    run_world(sim_est)
    run_world(sim_gt)
    loss.compute_loss(199)

# simulation engine with ground truth and estimated mass
sim_gt, sim_est = ParticleSimulator(), ParticleSimulator()
# mass_gt, mass_est = np.ones(sim_est.ngeom), np.array([0.8, 1])
mass_gt, mass_est = np.ones(sim_est.ngeom), np.ones(sim_est.ngeom)
sim_gt.geom_mass.from_numpy(mass_gt)
sim_est.geom_mass.from_numpy(mass_est)

loss = Loss((sim_est, sim_gt))

lr = 2e-3
max_iter = 200

### regress the mass based on position difference
for i in range(max_iter):
    sim_gt.initialize()
    sim_est.initialize()
    loss.clear_loss()

    # forward sims to compute loss and gradient
    with ti.Tape(loss.loss):
        forward()

    print('Iteration %d loss: %12.4f   mass_est: %12.4f   mass_gt: %8.4f   gradient: %.4f'%(
            i, loss.loss[None], sim_est.geom_mass.to_numpy()[0], mass_gt[0], sim_est.geom_mass.grad[0]))

    grad = sim_est.geom_mass.grad

    # update estimated mass
    mass_est -= lr * grad.to_numpy()
    sim_est.geom_mass.from_numpy(mass_est)


# ######################   render simulation  ######################
# ###  render one episode

# sim = ParticleSimulator()
# mass = np.array([0.8, 1])
# sim.geom_mass.from_numpy(mass)

# sim.initialize()
# for s in range(sim.max_step-1):
#     sim.collide(s)
#     sim.update(s)
#     sim.render(s)

