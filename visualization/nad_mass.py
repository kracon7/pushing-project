import os
import sys
import argparse
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation
from taichi_pushing.physics.block_object_util import BlockObject
import plotly.colors

shape_I = np.array([[0,0,1,1],
                    [0,1,1,2],
                    [0,2,1,3],
                    [0,3,1,4],
                    [0,4,1,5],
                    [0,5,1,6]])
shape_L = np.array([[0,0,1,1],
                    [1,0,2,1],
                    [2,0,3,1],
                    [0,1,1,2],
                    [0,2,1,3],
                    [0,3,1,4]])
shape_F = np.array([[0,0,1,1],
                    [0,1,1,2],
                    [1,1,2,2],
                    [0,2,1,3],
                    [0,3,1,4],
                    [1,3,2,4]])

method_offset = [0, 6, 12, 18, 24]


def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:
    
        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    if intermed <= 0 or len(colorscale) == 1:
        return colorscale[0][1]
    if intermed >= 1:
        return colorscale[-1][1]

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    # noinspection PyUnboundLocalVariable
    return plotly.colors.find_intermediate_color(
        lowcolor=low_color, highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb")
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot real robot force/torque and pose data')
    parser.add_argument('--data_dir', type=str, help='dir for force torque file and pose file')
    args = parser.parse_args()
    
    file_names = os.listdir(args.data_dir)
    viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
    colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=[0.75, 0.75, 0.75, 0.75, 0.75, 0.75],
    #                          y=[16, 13, 10, 7, 4.5, 1.5],
    #                          text=["I1", "I2", "L1", "L2", "F1", "F2"],
    #                          mode="text",
    #                         ))


    for i, f in enumerate(file_names):
        # off set of shapes
        if 'I1' in f:
            shape_offset = [2, 26.5]
            shape = shape_I.copy()
        elif 'I2' in f:
            shape_offset = [2, 20]
            shape = shape_I.copy()
        elif 'L1' in f:
            shape_offset = [2, 15.5]
            shape = shape_L.copy()
        elif 'L2' in f:
            shape_offset = [2, 11]
            shape = shape_L.copy()
        elif 'F1' in f:
            shape_offset = [2, 6.5]
            shape = shape_F.copy()
        elif 'F2' in f:
            shape_offset = [2, 2]
            shape = shape_F.copy()

        if "1" in f:
            gt = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        else:
            gt = np.array([0.4, 0.1, 0.1, 0.4, 0.4, 0.4])
        data = np.loadtxt(os.path.join(args.data_dir, f))[::2]
        diff = np.abs(data - gt)
        diff[0] *=1.8

        for j, s in enumerate(method_offset):
            for k, c in enumerate(shape):
                color = get_continuous_color(colorscale, diff[j, k]/0.4)
                x0 = shape_offset[0] + c[0] + s
                y0 = shape_offset[1] + c[1]
                x1 = shape_offset[0] + c[2] + s
                y1 = shape_offset[1] + c[3]
                fig.add_shape(type="rect",
                                x0=x0, y0=y0, x1=x1, y1=y1,
                                line=dict(
                                    color="RoyalBlue",
                                    width=1,
                                ),
                                fillcolor=color,
                            )
    
    fig.add_trace(go.Scatter(x=[0],
                             y=[0],
                            marker=dict(
                                size=16,
                                cmax=0.3,
                                cmin=0,
                                color=0,
                                colorbar=dict(
                                    title="Colorbar"
                                ),
                                colorscale="Viridis"
                            ),
                            mode="markers"))
    fig.update_xaxes(range=[0, 35], showgrid=False)
    fig.update_yaxes(range=[0, 35], showgrid=False)
    fig.update_layout(title="Kinematic Generation of a Planar Curve", 
                      hovermode="closest",
                      width = 1400,
                      height = 1400,
                      plot_bgcolor="#FFFFFF",
                      grid=None)
    fig.show()
