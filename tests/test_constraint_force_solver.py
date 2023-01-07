'''
Hidden state si loss contour and gradient direction plot with hidden state simulator
'''

import os
import argparse
import numpy as np
import sympy as sp
import plotly.figure_factory as ff
import plotly.graph_objects as go
import taichi as ti
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping
from taichi_pushing.physics.constraint_force_solver import ConstraintForceSolver
from taichi_pushing.optimizer.optim import Momentum
from taichi_pushing.physics.utils import Defaults

ti.init(arch=ti.cpu, debug=True)

def rotation_matrix(r: sp.Symbol):
    return sp.Matrix(2, 2, [sp.cos(r), -sp.sin(r), sp.sin(r), sp.cos(r)])

def cross_2d(v1: sp.Matrix, v2: sp.Matrix):
    return v1[0] * v2[1] - v1[1] * v2[0]

def right_orthogonal(v: sp.Matrix):
    return sp.Matrix(2, 1, [-v[1], v[0]])

def bottom_friction(gvel: list, geom_si: np.ndarray):
    # compute bottom friction force
    friction = []
    for i in range(len(gvel)):
        v = gvel[i]
        if sp.sqrt(v[0]**2 + v[1]**2) > 1e-6:
            f = geom_si[i] * v / sp.sqrt(v[0]**2 + v[1]**2)
        else:
            f = sp.Matrix(2, 1, [0, 0])
        friction.append(f.copy())
    return friction

def apply_external(geom_force: sp.Matrix, geom_torque: sp.Symbol, 
                        fx: sp.Symbol, fy: sp.Symbol, fw: sp.Symbol):
    geom_force = geom_force + sp.Matrix(2, 1, [fx, fy])
    geom_torque = geom_torque + fw
    return geom_force, geom_torque
          
def compute_ft(geom_force: list, geom_torque: list, gpos: list, qpos: sp.Matrix):
    body_force, body_torque = sp.Matrix(2, 1, [0, 0]), 0
    for i in range(len(geom_force)):
        body_force = body_force + geom_force[i]
        body_torque = body_torque + geom_torque[i]
        body_torque = body_torque + cross_2d(geom_force[i], qpos - gpos[i])
    return body_force, body_torque

def forward_body(qpos: sp.Matrix, rpos: sp.Symbol, qvel: sp.Matrix, rvel: sp.Symbol,
                 body_force: sp.Matrix, body_torque: sp.Symbol):
    qpos = qpos + DT * qvel
    rpos = rpos + DT * rvel
    qvel = qvel + DT * body_force / M
    rvel = rvel + DT * body_torque / I
    return qpos, rpos, qvel, rvel

def forward_geom(qpos: sp.Matrix, rpos: sp.Symbol, qvel: sp.Matrix, rvel: sp.Symbol):
    gpos, gvel = [], []
    for i in range(len(t0)):
        rot = rotation_matrix(rpos)
        gpos.append(qpos + rot * t0[i])
        gvel.append(qvel + rvel * right_orthogonal(rot * t0[i]))
    return gpos, gvel
     
def render(gpos: list):  # Render the scene on GUI
    temp = []
    for i in range(len(gpos)):
        temp.append([gpos[i][0], gpos[i][1]])
    np_pos = np.array(temp)
    np_pos = (np_pos - np.array([-0.4, -0.4])) / np.array([0.8, 0.8])

    # composite object
    r = 0.0125 * 800 / 0.6
    gui.circles(np_pos, color=0xffffff, radius=r)

    gui.show()


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_STEP = 50
GRAVITY = 9.8
DT = 0.01
param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
sim = HiddenStateSimulator(param_file)
gui = ti.GUI("sympy test", (800, 800))

mass = 0.1 * np.ones(sim.block_object.num_particle)
friction = 0.5 * np.ones(sim.block_object.num_particle)
n_partition = 6
mass_mapping = np.repeat(np.arange(n_partition), 
                            sim.ngeom//n_partition).astype("int")
friction_mapping = np.zeros(sim.block_object.num_particle).astype("int")

# Map to hidden state
hidden_state_mapping = HiddenStateMapping(sim)
hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                friction, friction_mapping)

print(hidden_state)         

# Canocical particle offset
t0 = []
for i in range(sim.block_object.num_particle):
    t0.append(sp.Matrix(2, 1, 
            sim.block_object.particle_coord[i] - hidden_state["body_com"]))

# Body mass and inertia
M, I = hidden_state["body_mass"], hidden_state["body_inertia"]
geom_si = hidden_state["composite_si"][hidden_state["si_mapping"]]

ngeom = len(t0)

# # Initial body qpos rpos qvel rvel
# qpos, qvel = sp.Matrix(2, 1, [0, 0]), sp.Matrix(2, 1, [0, 0])
# rpos, rvel = 0, 0

# for _ in range(100):
#     # Forward geom
#     gpos, gvel = forward_geom(qpos, rpos, qvel, rvel)

#     # render
#     render(gpos)

#     # Friction force on each particle
#     geom_force = bottom_friction(gvel, geom_si)
#     geom_torque = [0 for _ in range(ngeom)]

#     # Apply external force to particle i
#     geom_force[i], geom_torque[i] = apply_external(geom_force[i], geom_torque[i],
#                                                 0, 0, 3)

#     # Force torque on body level
#     body_force, body_torque = compute_ft(geom_force, geom_torque, gpos, qpos)

#     # Update body qpos rpos qvel rvel
#     qpos, rpos, qvel, rvel = forward_body(qpos, rpos, qvel, rvel,
#                                         body_force, body_torque)

# fix rotation wrt. particle 0
i = 10
x = sp.symbols('x:2')
qpos, qvel = sp.Matrix(2, 1, [0, 0]), sp.Matrix(2, 1, [x[0], x[1]])
rpos, rvel = 0, 1
gpos, gvel = forward_geom(qpos, rpos, qvel, rvel)
body_force, body_torque = sp.Matrix(2, 1, [0, 0]), 0
qpos1, rpos1, qvel1, rvel1 = forward_body(qpos, rpos, qvel, rvel,
                                    body_force, body_torque)
gpos1, gvel1 = forward_geom(qpos1, rpos1, qvel1, rvel1)
eq = sp.Eq(gpos1[i], gpos[i])
res = sp.solve([eq], x, dict=True)
sub = [(s, res[0][s]) for s in x]
print(res, "gpos0: ", gpos[i], 
        "  gpos1: ", gpos1[i].subs(sub))

# Fix the initial conditions from the solution
fx, fy, fw = sp.symbols('fx'), sp.symbols('fy'), sp.symbols('fw')
qpos, qvel = sp.Matrix(2, 1, [0, 0]), sp.Matrix(2, 1, [x[0], x[1]]).subs(sub)
rpos, rvel = 0, 1
gpos, gvel = forward_geom(qpos, rpos, qvel, rvel)
geom_force = bottom_friction(gvel, geom_si)
geom_torque = [0 for _ in range(ngeom)]
geom_force[i], geom_torque[i] = apply_external(geom_force[i], geom_torque[i],
                                                fx, fy, fw)
body_force, body_torque = compute_ft(geom_force, geom_torque, gpos, qpos)
qpos1, rpos1, qvel1, rvel1 = forward_body(qpos, rpos, qvel, rvel,
                                        body_force, body_torque)
qpos2, rpos2, qvel2, rvel2 = forward_body(qpos1, rpos1, qvel1, rvel1,
                                    sp.Matrix(2, 1, [0, 0]), 0)
gpos2, gvel2 = forward_geom(qpos2, rpos2, qvel2, rvel2)

diff1 = sp.simplify(gpos2[i] - gpos[i])
diff2 = sp.simplify(rvel1 - rvel)

D11, D12, D13 = sp.diff(diff1[0], fx), sp.diff(diff1[0], fy), sp.diff(diff1[0], fw)
D21, D22, D23 = sp.diff(diff1[1], fx), sp.diff(diff1[1], fy), sp.diff(diff1[1], fw)
D31, D32, D33 = sp.diff(diff2, fx), sp.diff(diff2, fy), sp.diff(diff2, fw)

D = sp.Matrix([[D11, D12, D13], [D21, D22, D23], [D31, D32, D33]])

res = sp.Matrix(3, 1, [0, 0, 10])
X = sp.Matrix(3, 1, [fx, fy, fw])
F = sp.Matrix(3, 1, [diff1[0], diff1[1], diff2])
for i in range(10):
    sub = [(fx, res[0]), (fy, res[1]), (fw, res[2])]
    print(F.subs(sub), X.subs(sub))
    res = X.subs(sub) - D.subs(sub).inv() * F.subs(sub)
    


# sim.input_parameters(hidden_state["body_mass"], 
#                      hidden_state["body_inertia"], 
#                      hidden_state["body_com"], 
#                      hidden_state["composite_si"], 
#                      hidden_state["si_mapping"], 
#                      {}, [])
# sim.run(SIM_STEP, render=True)