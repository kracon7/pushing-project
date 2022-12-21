'''
Example to the differentiation of taichi.abs(), especially the subgradient at x=0.

Given:
x : array [-1, 0, 1]
y = taichi.abs(x)
loss = y.sum()

Goal:
Compute the gradient d(y) / d(x)
'''
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class TaichiTest:
    def __init__(self) -> None:
        self.x = ti.Vector.field(3, ti.f64, shape=(), needs_grad=True)
        self.x[None] = [-1, 0, 1]
        self.y = ti.Vector.field(3, ti.f64, shape=(), needs_grad=True)
        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)

    @ti.kernel
    def comppute_loss(self):
        self.y[None] = ti.abs(self.x[None])
        self.loss[None] = self.y[None].sum()

if __name__ == '__main__':
    taichi_test = TaichiTest()

    with ti.ad.Tape(taichi_test.loss, validation=True):
        taichi_test.comppute_loss()
    print("Loss: %.5f  Gradient: "%(taichi_test.loss[None]), taichi_test.x.grad)
