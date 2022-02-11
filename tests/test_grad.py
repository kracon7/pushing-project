import taichi as ti

ti.init()

x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
y = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
b = ti.field(dtype=ti.f32, shape=())
loss = ti.field(ti.f32, shape=(), needs_grad=True)

@ti.kernel
def compute_loss():
    y[None] = ti.sin(x[None])
    loss[None] = y[None] + b[None]

with ti.Tape(loss):
    compute_loss()

print('dy/dx =', x.grad[None], ' at x =', x[None])