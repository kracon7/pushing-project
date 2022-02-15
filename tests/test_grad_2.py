import numpy as np
import taichi as ti

ti.init()

x = ti.field(dtype=ti.f32, shape=3, needs_grad=True)
y = ti.field(dtype=ti.f32, shape=3, needs_grad=True)
idx = ti.field(dtype=ti.i32, shape=3)

x.from_numpy(np.array([1,2,3]))
y.from_numpy(np.array([1,1,1]))
idx.from_numpy(np.array([0,1,2]))

loss = ti.field(ti.f32, shape=(), needs_grad=True)
loss[None] = 0

@ti.kernel
def compute_loss():
    for i in idx:
        if x[i] != y[i]:
            loss[None] += y[i] + x[i]

with ti.Tape(loss):
    compute_loss()

print('dy/dx =', x.grad, ' at x =', x)