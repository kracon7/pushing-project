import numpy as np
import taichi as ti

ti.init(debug=True)

@ti.data_oriented
class TestClass:
    def __init__(self):  # Initializer of the pushing environment
        self.sum = ti.field(ti.f32, 2)
        self.a = ti.field(ti.f32, shape=10)
        self.b = np.arange(10).astype('int')
        self.idx = ti.field(ti.i32, shape=10)
        self.idx.from_numpy(self.b)

    @ti.kernel
    def add_values(self):
    	for i in range(10):
    		print(self.idx[i])
    		self.sum[0] += self.a[self.idx[i]]

    @ti.kernel
    def test(self):
    	self.add_values()

c = TestClass()
# c.test()
c.add_values()
        