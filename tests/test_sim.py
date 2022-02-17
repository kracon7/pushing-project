import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.composite_util import Composite2D
from taichi_pushing.pushing_simulator import PushingSimulator

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':
    composite = Composite2D(5)
    sim = PushingSimulator(composite)

    sim.composite_mass.from_numpy(composite.mass_dist)

    sim.clear_all()
    sim.initialize(15)

    for s in range(sim.max_step-1):

        for e in sim.gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()

        sim.collide(s)
        sim.compute_ft(s)
        sim.update(s)
        sim.render(s)

        print(sim.body_force[s, 0], sim.body_force[s, 1])