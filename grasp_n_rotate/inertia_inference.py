import os
import argparse
import numpy as np
import taichi as ti
from taichi_pushing.physics.hidden_state_simulator import HiddenStateSimulator
from taichi_pushing.physics.hidden_state_mapping import HiddenStateMapping

ti.init(arch=ti.cpu, debug=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@ti.data_oriented
class InertiaInference:
    def __init__(self, batch_size=3):
        self.batch_size = batch_size
        self.ij = ti.field(ti.f64, batch_size)
        self.px = ti.field(ti.f64, batch_size)
        self.py = ti.field(ti.f64, batch_size)
        self.M = ti.field(ti.f64, shape=())
        self.icm = ti.field(ti.f64, shape=(), needs_grad=True)
        self.cx = ti.field(ti.f64, shape=(), needs_grad=True)
        self.cy = ti.field(ti.f64, shape=(), needs_grad=True)
        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)
        
    @ti.kernel
    def add_loss(self, b: ti.i32):
        self.loss[None] += ti.abs(self.ij[b] - self.icm[None] - \
                                  self.M[None] * (self.cx - self.px[b])**2 - \
                                  self.M[None] * (self.cy - self.py[b])**2)

    def comppute_loss(self):
        for i in range(self.batch_size):
            self.add_loss(i)

if __name__ == '__main__':
    inertia_inference = InertiaInference()
    
    parser = argparse.ArgumentParser(description='System ID on block object model')
    parser.add_argument("--data_dir", type=str, default='data/block_object_L_1')
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip('/')
    data_suffix = data_dir.split('/')[-1].split('_')[-1]
    param_file = data_dir.split('/')[-1].rstrip('_' + data_suffix) + '.yaml'
    print("Loading data batch %s for %s"%(data_suffix, param_file))
    param_file = os.path.join(ROOT, 'config', param_file)
    sim = HiddenStateSimulator(param_file)

    with ti.ad.Tape(taichi_test.loss, validation=True):
        taichi_test.comppute_loss(mean=True)
    print("When mean is True, Loss: %.5f  Gradient: "%(taichi_test.loss[None]), taichi_test.x.grad)

    with ti.ad.Tape(taichi_test.loss, validation=True):
        taichi_test.comppute_loss(mean=False)
    print("When mean is False, Loss: %.5f  Gradient: "%(taichi_test.loss[None]), taichi_test.x.grad)
