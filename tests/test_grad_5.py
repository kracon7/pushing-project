import os
import sys
import numpy as np
import taichi as ti

ti.init()

DTYPE = ti.f32

@ti.data_oriented
class ExampleClass:
    def __init__(self):
      
        k = 2
        
        # mass of each circle
        self.x = ti.field(DTYPE, k, needs_grad=True)
        self.x.from_numpy(np.ones(k))
      
        # pos, vel and force of each circle
        self.y = ti.Vector.field(2, DTYPE, shape=(2, k), needs_grad=True)
 
        self.z = ti.Vector.field(2, DTYPE, shape=(), needs_grad=True)

        self.m = ti.field(DTYPE, (), needs_grad=True)
        self.n = ti.field(DTYPE, (), needs_grad=True)

    @ti.kernel
    def test(self):
        self.m[None] += self.x[0]
        self.z[None] += self.x[0] * self.m[None] * self.y[0, 0]

        self.n[None] += self.x[0] * (self.y[0, 0] - self.z[None]).norm()**2
        
        # # change norm() to norm(1e-5) and the gradient is no longer NaN
        # self.n[None] += self.x[0] * (self.y[0, 0] - self.z[None]).norm(1e-5)**2
            
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
    def compute_loss(self):
        self.loss[None] = self.engines[0].y[1, 0][0] - self.engines[1].y[1, 0][0]
                              

sim1 = ExampleClass()
sim2 = ExampleClass()
loss = Loss((sim1, sim2))

def forward():
    sim1.test()
    sim2.test()
    loss.compute_loss()

with ti.Tape(loss.loss):
    forward()

print(loss.loss[None])
print(sim1.x.grad)
