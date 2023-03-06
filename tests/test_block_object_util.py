import os
import numpy as np
import plotly.graph_objects as go
from taichi_pushing.physics.block_object_util import BlockObject

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
param_file = os.path.join(ROOT, 'config', 'hammer.yaml')
block_object = BlockObject(param_file)
coord = block_object.particle_coord
print(block_object.mass_mapping)

fig = go.Figure(
        data = go.Scatter(x=coord[:, 0], y=coord[:, 1], mode = 'markers'),
        layout=go.Layout(
            xaxis=dict(range=[0.05, 0.65], autorange=False, title="X"),
            yaxis=dict(range=[-0.3, 0.3], autorange=False, title="Y"),
            height=800, width=800
    )
)
fig.show()