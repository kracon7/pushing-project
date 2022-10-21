import numpy as np
import taichi as ti

ti.init()

x = ti.field(dtype=ti.f32, shape=3, needs_grad=True)
y = ti.field(dtype=ti.f32, shape=3, needs_grad=True)
idx = ti.field(dtype=ti.i64, shape=3)

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

with ti.ad.Tape(loss):
    compute_loss()
    compute_loss()

    # # Only use ti.kernel with Tape for auto-differentiation
    # # Uncomment this line and the gradient dy/dx remains the same
    # loss[None] += y[0] + x[0] + y[1] + x[1]

print('loss = ', loss[None], ' dy/dx =', x.grad.to_numpy(), ' at x =', x)