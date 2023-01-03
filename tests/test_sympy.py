import os
import numpy as np
import sympy as sp
import taichi as ti
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping

ti.init(arch=ti.cpu, debug=True)

# mappings
mass_mapping = np.array([0,0,1,1,2,2,3,3,4,4,5,5]).astype("int")
friction_mapping = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).astype("int")

n_particle = mass_mapping.shape[0]
n_mass, n_mu = len(set(mass_mapping)), len(set(friction_mapping))

m2mu = np.array([0,0,0,0,0,0]).astype("int")

composite_mass = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
composite_friction = np.array([0.8, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
particle_pos = np.array([[0,-1],
                         [0,1],
                         [1,-1],
                         [1,1],
                         [2,-1],
                         [2,1],
                         [3,-1],
                         [3,1],
                         [4,-1],
                         [4,1],
                         [5,-1],
                         [5,1]])

m = sp.symbols('m:%d'%n_mass)
mu = sp.symbols('mu:%d'%n_mu)

# expressions for total mass, center of mass and moment of inertia
M = 0
for i in range(n_particle):
    M = M + m[mass_mapping[i]]

cx, cy = 0, 0
for i in range(n_particle):
    cx = cx + particle_pos[i,0] * m[mass_mapping[i]] / M
    cy = cy + particle_pos[i,1] * m[mass_mapping[i]] / M

I = 0
for i in range(n_particle):
    I = I + m[mass_mapping[i]] * ((particle_pos[i,0] - cx)**2 + (particle_pos[i,1] - cy)**2)

si = []
for i in range(n_mass):
    si.append(9.8 * m[i] * mu[m2mu[i]])

# Forward mapping
sub = []
for i in range(n_mass):
    sub.append((m[i], composite_mass[i]))
for i in range(n_mu):
    sub.append((mu[i], composite_friction[i]))

x = sp.symbols('x:%d'%(4+n_mass))
equations = []
equations.append(sp.Eq(x[0], M.subs(sub)))
equations.append(sp.Eq(x[1], cx.subs(sub)))
equations.append(sp.Eq(x[2], cy.subs(sub)))
equations.append(sp.Eq(x[3], I.subs(sub)))
for i in range(n_mass):
    equations.append(sp.Eq(x[i+4], si[i].subs(sub)))
res = sp.solve(equations, x, dict=True)
print(res)

# Backward mapping
h = np.array([res[0][x[i]] for i in range(n_mass+4)])
equations = []
equations.append(sp.Eq(M, h[0]))
equations.append(sp.Eq(cx, h[1]))
equations.append(sp.Eq(cy, h[2]))
equations.append(sp.Eq(I, h[3]))
for i in range(n_mass):
    equations.append(sp.Eq(si[i], h[i+4]))
res = sp.solve(equations, m+mu, dict=True)
print(res)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
sim = HiddenStateSimulator(param_file)
mapper = HiddenStateMapping(sim)
mapper.particle_coord = particle_pos
mapper.n_particle = n_particle

hidden_state = mapper.map_to_hidden_state(composite_mass, mass_mapping, 
                                    composite_friction, friction_mapping)
print(hidden_state)
explicit_state = mapper.map_to_explicit_state(hidden_state, mass_mapping, friction_mapping)
print(explicit_state)