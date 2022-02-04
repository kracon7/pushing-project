'''
Testing for Compoite2D and PushingSimulator
'''
import os
import sys
import numpy as np
import taichi as ti

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from sim2d.pushing_simulator import PushingSimulator
from sim2d.composite_util import Composite2D

composite = Composite2D(0)
sim = PushingSimulator(composite)