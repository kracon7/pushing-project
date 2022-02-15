# iteration through list in taichi kernel causes error
import numpy as np
import taichi as ti

ti.init()

l = [0, 1, 3]
a = ti.field(ti.f32, shape=5)
a.from_numpy(np.arange(5))

s = ti.field(ti.f32, shape=())

@ti.kernel
def test():
	for i in l:
		s += a[i]

test()
print(s[None])
