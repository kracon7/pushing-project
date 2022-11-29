import os
import math
import taichi as ti

class Defaults:
    """Aggregates general simulation parameters defaults.
    """
    # Dimensions
    DIM = 2

    FPS = 100
    DT = 1.0 / FPS
    DTYPE = ti.f64
    DEVICE = ti.cpu

    def __init__(self):
        pass
