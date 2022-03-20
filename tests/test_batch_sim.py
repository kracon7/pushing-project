# import taichi as ti
# import numpy as np

# ti.init()

# def render(gui):  # Render the scene on GUI
#     gui.circles(np.array([[0.5, 0.5]]), color=0x09ffff, radius=100)
#     gui.show()

# g1 = ti.GUI('gui_1', (800, 800))
# g2 = ti.GUI('gui_2', (800, 800))

# for i in range(10000):
#     render(g1)
#     render(g2)

import os
import sys
import numpy as np
import taichi as ti
from taichi_pushing.physics.composite_util import Composite2D
from taichi_pushing.physics.batch_pushing_simulator import PushingSimulator

ti.init(arch=ti.cpu, debug=True)

if __name__ == '__main__':
    composite = Composite2D(2)
    sim = PushingSimulator(composite, bs=2)

    sim.composite_mass.from_numpy(composite.mass_dist)

    sim.clear_all()
    sim.initialize([90, 0])

    for s in range(sim.max_step-1):

        # for e in sim.gui.get_events(ti.GUI.PRESS):
        #     if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
        #         exit()

        sim.collide(s)
        sim.compute_ft(s)
        sim.update(s)
        sim.render(s)