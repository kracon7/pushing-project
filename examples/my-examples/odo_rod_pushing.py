'''
Composite object pushing example
Interactions considered include spring force, damping force and friction force
'''
import os
import sys
import numpy as np
import taichi as ti

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from sim2d.composite_util import Composite2D

ti.init(arch=ti.gpu, debug=True)

@ti.data_oriented
class PushingSimulator:
    def __init__(self, dt=1e-3):  # Initializer of the pushing environment
        # number of particles in the rod
        N = 10
        self.N = N 
        self.dt = dt
        self.rod_radius = 0.01
        self.hand_radius = 0.03

        self.ks = 1e4
        self.eta = 10
        self.mu = 0.1

        # center of the screen
        self.center = ti.Vector.field(2, ti.f32, ())

        # pos, vel and force of each particle
        self.pos = ti.Vector.field(2, ti.f32, N+1)
        self.vel = ti.Vector.field(2, ti.f32, N+1)
        self.force = ti.Vector.field(2, ti.f32, N+1)

        self.body_qpos = ti.Vector.field(2, ti.f32, 2)
        self.body_qvel = ti.Vector.field(2, ti.f32, 2)
        self.body_rpos = ti.Vector.field(2, ti.f32, ())
        self.body_rvel = ti.Vector.field(2, ti.f32, ())

        self.geom_qpos0 = ti.Vector.field(2, ti.f32, N)

        # net force and torque on body aggregated from all particles
        self.body_force = ti.Vector.field(2, ti.f32, 2)
        self.body_torque = ti.field(ti.f32, 2)

        # radius and mass of the particles
        self.radius = ti.field(ti.f32, N+1)
        self.mass = ti.field(ti.f32, N+1)
        for i in range(N):
            self.radius[i] = self.rod_radius
            self.mass[i] = 0.5
        self.radius[N] = self.hand_radius
        self.mass[N] = 2

        self.total_mass, self.inertia = ti.field(ti.f32, ()), ti.field(ti.f32, ())

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
    def compute_force(self):
        # compute particle force based on pair-wise collision
        # clear force
        for i in range(self.N+1):
            self.force[i] = [0.0, 0.0]

        for i in range(self.N):
            p = self.pos[i]
            
            r_ij = self.pos[self.N] - p
            r = r_ij.norm(1e-5)
            n_ij = r_ij / r
            if (self.radius[i] + self.radius[self.N] - r) > 0:
                fs = - self.ks * (self.radius[i] + self.radius[self.N] - r) * n_ij  # spring force
                self.force[i] += fs

                v_ij = self.vel[self.N] - self.vel[i]   # relative velocity
                vn_ij = v_ij.dot(n_ij) * n_ij  # normal velocity
                fb = self.eta * vn_ij   # damping force
                self.force[i] += fb

                vt_ij = v_ij - vn_ij   # tangential velocity
                if vn_ij.norm() > 1e-4:
                    ft = self.mu * (fs.norm() + fb.norm()) * vt_ij / vn_ij.norm()
                    self.force[i] += ft
                # force[N] -= fs

    @ti.kernel
    def apply_external(self, geom_id: ti.i32, fx: ti.f32, fy: ti.f32):
        self.force[geom_id][0] += fx
        self.force[geom_id][1] += fy

    @ti.kernel
    def compute_ft(self):

        # clear net force and torque
        self.body_force[0], self.body_force[1], self.body_torque[0], self.body_torque[1] = [0.,0.], [0.,0.], 0., 0.
        for i in range(self.N):
            self.body_force[0] += self.force[i]
            self.body_torque[0] += self.cross_2d(self.force[i], (self.body_qpos[0] - self.pos[i]))

        self.body_force[1] += self.force[self.N]

    @ti.kernel
    def update(self):
        self.body_qvel[0] += self.dt * self.body_force[0] / self.total_mass[None]
        self.body_rvel[None][0] += self.dt * self.body_torque[0] / self.inertia[None]
        self.body_qvel[1] += self.dt * self.body_force[1] / self.mass[self.N]

        # update body qpos and rpos
        self.body_qpos[0] += self.dt * self.body_qvel[0] 
        self.body_rpos[None][0] += self.dt * self.body_rvel[None][0]
        self.body_qpos[1] += self.dt * self.body_qvel[1]

        for i in range(self.N):
            rot = self.rotation_matrix(self.body_rpos[None][0])
            self.pos[i] = self.body_qpos[0] + rot @ self.geom_qpos0[i]
            self.vel[i] = self.body_qvel[0] + self.body_rvel[None][0] * self.right_orthogonal(rot @ self.geom_qpos0[i])

        self.pos[self.N] = self.body_qpos[1]
        self.vel[self.N] = self.body_qvel[1]

    @ti.kernel
    def initialize(self):
        self.center[None] = [0.5, 0.5]
        rod_dp = ti.Vector([0.02, 0])
        for i in range(self.N):
            self.pos[i] = self.center[None] + i * rod_dp
            self.vel[i] = [0., 0.]
            self.force[i] = [0., 0.]

        offset = ti.Vector([0, -0.05])
        self.pos[self.N] = self.center[None] + offset
        self.vel[self.N] = [0., 0.]
        self.force[self.N] = [0., 0.]

        # total mass of the rod
        for i in range(self.N):
            self.total_mass[None] += self.mass[i]

        # compute the body_qpos, body_qvel, body_rpos, body_rvel
        self.body_qpos[0], self.body_qpos[1] = [0., 0.], [0., 0.]
        for i in range(self.N):
            self.body_qpos[0] += self.mass[i] / self.total_mass[None] * self.pos[i]

        self.body_qpos[1] = self.pos[self.N]

        self.body_qvel[0], self.body_qvel[1] = [0., 0.], [0., 0.]

        self.body_rpos[None] = [0., 0.]
        self.body_rvel[None] = [0., 0.]

        # compute geom_qpos0
        for i in range(self.N):
            self.geom_qpos0[i] = self.pos[i] - self.body_qpos[0]

            # compute the force and torque from particle forces
            self.inertia[None] += self.mass[i] * (self.pos[i] - self.body_qpos[0]).norm()**2

    def render(self, gui):  # Render the scene on GUI
        np_pos = self.pos.to_numpy()
        gui.circles(np_pos[:self.N], color=0xffffff, radius=self.rod_radius*800)
        gui.circles(np_pos[self.N].reshape(1,2), color=0x09ffff, radius=self.hand_radius*800)
        

gui = ti.GUI('Rod Pushing', (800, 800))
sim = PushingSimulator()
sim.initialize()
while gui.running:

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        
    sim.compute_force()
    sim.apply_external(sim.N, 5, 1)
    sim.compute_ft()
    sim.update()
    sim.render(gui)
    gui.show()
