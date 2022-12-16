'''
Example to show why averaging the accumulated loss must pre-compute the
coeficient of the batch size, instead of adding followed by division.

Given:
x : array [0, 1, 2]
if mean is True
    loss = sum(x)
else
    loss = sum(x) / size(x)

Goal:
Compute the gradient d(loss) / d(x)
'''
import numpy as np
import taichi as ti

ti.init(arch=ti.cpu, debug=True)

@ti.data_oriented
class TaichiTest:
    def __init__(self) -> None:
        self.n = 3
        self.x = ti.field(ti.f64, self.n, needs_grad=True)
        self.a = ti.field(ti.f64, shape=())
        self.loss = ti.field(ti.f64, shape=(), needs_grad=True)
        
    @ti.kernel
    def add_loss(self, i: ti.i32):
        self.loss[None] += self.a[None] * self.x[i]

    @ti.kernel
    def average_loss(self, b: ti.f64):
        self.loss[None] /= b

    def comppute_loss(self, mean=True):
        for i in range(self.n):
            self.add_loss(i)
        if mean:
            self.average_loss(self.n)

if __name__ == '__main__':
    taichi_test = TaichiTest()
    taichi_test.x.from_numpy(np.arange(taichi_test.n))
    taichi_test.a[None] = 1

    with ti.ad.Tape(taichi_test.loss, validation=True):
        taichi_test.comppute_loss(mean=True)
    print("When mean is True, Loss: %.5f  Gradient: "%(taichi_test.loss[None]), taichi_test.x.grad)

    with ti.ad.Tape(taichi_test.loss, validation=True):
        taichi_test.comppute_loss(mean=False)
    print("When mean is False, Loss: %.5f  Gradient: "%(taichi_test.loss[None]), taichi_test.x.grad)
