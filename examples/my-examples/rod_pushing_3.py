'''
rod pushing example
Consider the spring force, damping force, side friction force and bottom friction froce
'''

import numpy as np
import taichi as ti

ti.init(arch=ti.gpu, debug=True)

# global control
paused = ti.field(ti.i32, ())

# number of particles in the rod
N = 10 
rod_radius = 0.01
hand_radius = 0.03

ks = 1e4
eta = 10
mu_s = 0.2
mu_b = 1


# center of the screen
center = ti.Vector.field(2, ti.f32, ())

# pos, vel and force of each particle
# Nx2 vectors
pos = ti.Vector.field(2, ti.f32, N+1)
vel = ti.Vector.field(2, ti.f32, N+1)
force = ti.Vector.field(2, ti.f32, N+1)

body_qpos = ti.Vector.field(2, ti.f32, 2)
body_qvel = ti.Vector.field(2, ti.f32, 2)
body_rpos = ti.field(ti.f32, 2)
body_rvel = ti.field(ti.f32, 2)

geom_qpos0 = ti.Vector.field(2, ti.f32, N)

# net force and torque on body aggregated from all particles
body_force = ti.Vector.field(2, ti.f32, 2)
body_torque = ti.field(ti.f32, 2)

# radius and mass of the particles
radius = ti.field(ti.f32, N+1)
mass = ti.field(ti.f32, N+1)
for i in range(N):
    radius[i] = rod_radius
    mass[i] = 0.5
radius[N] = hand_radius
mass[N] = 2

total_mass, inertia = ti.field(ti.f32, ()), ti.field(ti.f32, ())

dt = 1e-3

@ti.func
def rotation_matrix(r):
    return ti.Matrix([[ti.cos(r), -ti.sin(r)], [ti.sin(r), ti.cos(r)]])

@ti.func
def cross_2d(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]

@ti.func
def right_orthogonal(v):
    return ti.Vector([-v[1], v[0]])

@ti.kernel
def compute_force():
    # compute particle force based on pair-wise collision
    # clear force
    for i in range(N+1):
        force[i] = [0.0, 0.0]

    for i in range(N):
        p = pos[i]
        
        r_ij = pos[N] - p
        r = r_ij.norm(1e-5)
        n_ij = r_ij / r
        if (radius[i] + radius[N] - r) > 0:
            fs = - ks * (radius[i] + radius[N] - r) * n_ij  # spring force
            force[i] += fs

            v_ij = vel[N] - vel[i]   # relative velocity
            vn_ij = v_ij.dot(n_ij) * n_ij  # normal velocity
            fb = eta * vn_ij   # damping force
            force[i] += fb

            vt_ij = v_ij - vn_ij   # tangential velocity

            # side friction is activated with non-zero tangential velocity and non-breaking contact
            if vn_ij.norm() > 1e-4 and v_ij.dot(n_ij) < -1e-4:
                ft = mu_s * (fs.norm() + fb.norm()) * vt_ij / vn_ij.norm()
                force[i] += ft
            
        # bottom friction
        if vel[i].norm() > 1e-5:
            fb = - mu_b * mass[i] * (vel[i] / vel[i].norm())
            force[i] += fb


@ti.kernel
def apply_external(geom_id: ti.i32, fx: ti.f32, fy: ti.f32):
    force[geom_id][0] += fx
    force[geom_id][1] += fy

@ti.kernel
def compute_ft():

    # clear net force and torque
    body_force[0], body_force[1], body_torque[0], body_torque[1] = [0.,0.], [0.,0.], 0., 0.
    for i in range(N):
        body_force[0] += force[i]
        body_torque[0] += cross_2d(force[i], (body_qpos[0] - pos[i]))

    body_force[1] += force[N]

@ti.kernel
def update():
    body_qvel[0] += dt * body_force[0] / total_mass[None]
    body_rvel[0] += dt * body_torque[0] / inertia[None]
    body_qvel[1] += dt * body_force[1] / mass[N]

    # print(body_qvel[1], body_qpos[1])

    # update body qpos and rpos
    body_qpos[0] += dt * body_qvel[0] 
    body_rpos[0] += dt * body_rvel[0]
    body_qpos[1] += dt * body_qvel[1]

    for i in range(N):
        rot = rotation_matrix(body_rpos[0])
        pos[i] = body_qpos[0] + rot @ geom_qpos0[i]
        vel[i] = body_qvel[0] + body_rvel[0] * right_orthogonal(rot @ geom_qpos0[i])

    pos[N] = body_qpos[1]
    vel[N] = body_qvel[1]

@ti.kernel
def initialize():
    center[None] = [0.5, 0.5]
    rod_dp = ti.Vector([0.02, 0])
    for i in range(N):
        pos[i] = center[None] + i * rod_dp
        vel[i] = [0., 0.]
        force[i] = [0., 0.]

    offset = ti.Vector([0, -0.05])
    pos[N] = center[None] + offset
    vel[N] = [0., 0.]
    force[N] = [0., 0.]

    # total mass of the rod
    for i in range(N):
        total_mass[None] += mass[i]

    # compute the body_qpos, body_qvel, body_rpos, body_rvel
    body_qpos[0], body_qpos[1] = [0., 0.], [0., 0.]
    for i in range(N):
        body_qpos[0] += mass[i] / total_mass[None] * pos[i]

    body_qpos[1] =  pos[N]

    body_qvel[0], body_qvel[1] = [0., 0.], [0., 0.]

    body_rpos, body_rvel = [0., 0.], [0., 0.]

    # compute geom_qpos0
    for i in range(N):
        geom_qpos0[i] = pos[i] - body_qpos[0]

        # compute the force and torque from particle forces
        inertia[None] += mass[i] * (pos[i] - body_qpos[0]).norm()**2



gui = ti.GUI('Rod Pushing', (800, 800))

initialize()
s = 0
while gui.running:

    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]

    if not paused[None]:

        compute_force()
        if s < 100:
            apply_external(N, 0, 5)
        compute_ft()
        update()
        # initialize()

    np_pos = pos.to_numpy()
    gui.circles(np_pos[:N], color=0xffffff, radius=rod_radius*800)
    gui.circles(np_pos[N].reshape(1,2), color=0x09ffff, radius=hand_radius*800)
    gui.show()

    s += 1