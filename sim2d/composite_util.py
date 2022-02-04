import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def plot_polygon(ax, coord, gt_mass_regions=None):
    '''
    Plot a polygon from 2D coordinates
    Args:
        coord -- ndarray, shape (N,2)
        gt_mass_regions -- ground truth mass defined by rectangles
    '''
    ax.plot(coord[:, 0], coord[:,1], color='deepskyblue')
    ax.plot([coord[0,0], coord[-1,0]], 
             [coord[0,1], coord[-1,1]], color='deepskyblue')
    ax.axis('equal')
    
    if gt_mass_regions:
        for i, region in enumerate(gt_mass_regions):
            rc = Rectangle((region[0], region[1]), region[2] - region[0], 
                            region[3] - region[1])
            pc = PatchCollection([rc], facecolor=color_bar[i], alpha=0.5,
                                 edgecolor=None)
            ax.add_collection(pc)
    plt.show()

def voxelize(polygon, gt_mass, voxel_size):
    '''
    Map the object mass regions to a grid
    Args:
        polygon -- 2D coordinates of vertex, shape (N, 2)
        gt_mass -- region and density, 
                   [bottom_left_x, bottom_left_y, top_right_x, top_right_y, density]
        voxel_size -- voxel size, float
    '''
    xmax, xmin, ymax, ymin = polygon[:,0].max(), polygon[:,0].min(), polygon[:,1].max(), polygon[:,1].min()
    xx, yy = np.arange(xmin, xmax+voxel_size, voxel_size), np.arange(ymin, ymax+voxel_size, voxel_size)
    
    xx, yy = np.meshgrid(xx, yy)
    p = path.Path(polygon)
    flags = p.contains_points(np.hstack((xx.flatten()[:,np.newaxis],
                                         yy.flatten()[:,np.newaxis])),
                              radius = -0.01
                             ).reshape(xx.shape)
    
    pts = np.hstack((xx.flatten()[:,np.newaxis], 
                     yy.flatten()[:,np.newaxis]))[flags.flatten()]
    
    voxel = []
    for pt in pts:
        x, y = pt
        for region in gt_mass:
            if x>=region[0] and x<=region[2] and y>=region[1] and y<=region[3]:
                voxel.append(np.array([x, y, region[4]]))
                break
    voxel = np.stack(voxel)
    
    # fig, ax = plt.subplots(1,1)
    # ax.plot(polygon[:, 0], polygon[:,1], color='deepskyblue')
    # ax.plot([polygon[0,0], polygon[-1,0]], 
    #          [polygon[0,1], polygon[-1,1]], color='deepskyblue')
    # ax.axis('equal')
    
    # if gt_mass:
    #     for i, region in enumerate(gt_mass):
    #         rc = Rectangle((region[0], region[1]), region[2] - region[0], 
    #                         region[3] - region[1])
    #         pc = PatchCollection([rc], facecolor=color_bar[i], alpha=0.2,
    #                              edgecolor=None)
    #         ax.add_collection(pc)
    # for pt in voxel:
    #     ax.plot(pt[0], pt[1], c=[pt[2]*90/255, 0, 0], marker='o')
    # plt.show()
    return voxel

def remap(coarse, voxel_size, fine):
    '''
    index remap from fine voxel coordinate to coarse voxel coordinate
    Args:
        coarse -- 2D coordinates, shape (N, 2)
        voxel_size -- voxel size of coarse voxel
        fine -- 2D coordinates, shape (M, 2)
    '''
    point_tree = spatial.cKDTree(coarse)
    neighbors_list = point_tree.query_ball_point(fine, 2*voxel_size)
    
    mapping = []
    for i, pt in enumerate(fine):
        neighbors = neighbors_list[i]
        dist = np.linalg.norm(coarse[neighbors] - fine[i], axis=1)
        mapping.append(neighbors[np.argmin(dist)])

    return np.stack(mapping).astype('int')

object_names = ['hammer', 'drill', 'pan', 'spray_bottle', 'wrench']
color_bar = ["orange","pink","blue","brown","red","grey","yellow","green"]

poly_coords = {'hammer': np.array([[0., 0.],
                                   [0, 7], 
                                   [2, 9],
                                   [2, 5], 
                                   [15, 5], 
                                   [15, 3], 
                                   [2, 3],
                                   [2, 0]
                                  ]),
               'drill': np.array([[2, 0],
                                  [0, 2],
                                  [0, 9],
                                  [1, 10],
                                  [5, 10],
                                  [6, 9],
                                  [6, 8],
                                  [11.7, 8],
                                  [11.7, 10],
                                  [15, 10],
                                  [15, 3],
                                  [11.7, 3],
                                  [11.7, 5],
                                  [6, 5],
                                  [6, 2],
                                  [4, 0]
                                 ]),
               'pan': np.array([[6.46715673, -2.67878403],  
                                [4.94974747, -4.94974747],
                                [2.67878403, -6.46715673],
                                [0., -7],
                                [-2.67878403, -6.46715673],
                                [-4.94974747, -4.94974747], 
                                [-6.46715673, -2.67878403],
                                [-7.,     0],
                                [-6.46715673,  2.67878403],
                                [-4.94974747, 4.94974747],
                                [-2.67878403, 6.46715673],
                                [-0. ,    7],     
                                [2.67878403, 6.46715673],
                                [4.94974747, 4.94974747],
                                [6.46715673, 2.67878403],
                                [6.7,  1.5], 
                                [15, 1.5],
                                [15, -1.5],
                                [6.7, -1.5]
                               ]),
               'spray_bottle': np.array([[1, 0],
                                         [0, 1],
                                         [0, 5],
                                         [1, 6],
                                         [2, 7],
                                         [2, 10],
                                         [1, 10],
                                         [1, 11.5],
                                         [7, 11.5],
                                         [7, 10],
                                         [5.5, 10],
                                         [5.5, 8],
                                         [4.5, 10],
                                         [4, 10],
                                         [4, 7],
                                         [5, 6],
                                         [6, 5],
                                         [6, 1],
                                         [5, 0]
                                        ]),
               'wrench': np.array([[2.5,0],
                                   [2, 0.5],
                                   [2, 10],
                                   [1, 10.5],
                                   [0.3, 12],
                                   [0.4, 13],
                                   [1, 14],                                   
                                   [1.5, 14],
                                   [1.5, 12],
                                   [3, 11],
                                   [4.5, 12],
                                   [4.5, 14],
                                   [5, 14],
                                   [5.6, 13],
                                   [5.7, 12],
                                   [5, 10.5],
                                   [4, 10],
                                   [4, 0.5],
                                   [3.5, 0]
                                  ])
              }
gt_mass_dist = {'hammer': [[  0,    0,    2,   9,    2], 
                           [  2,    3,   15,   5,  0.5]],
                'drill':  [[  0,    0,    6,  10,  1.8], 
                           [  6,    5,   11.7,   8,  0.3],
                           [ 11.7,    3,   15,  10,  1.1]],
                'pan':    [[ -7,   -7,  6.7,   7,  1.3],
                           [6.7, -1.5,   15, 1.5,  0.4]],
                'spray_bottle': [[0, 0, 6, 7, 1],
                                 [0, 7, 7, 11.5, 0.4]],
                'wrench': [[0, 0, 6, 14, 2]]
               }


class Composite2D():
    """composite 2d object"""
    def __init__(self, obj_idx, coarse_vsize=0.8, fine_vsize=0.2):
        obj_name = object_names[obj_idx]
        self.polygon = poly_coords[obj_name]
        self.C = voxelize(poly_coords[obj_name], gt_mass_dist[obj_name], coarse_vsize)
        self.F = voxelize(poly_coords[obj_name], gt_mass_dist[obj_name], fine_vsize)
        self.mapping = remap(self.C[:,:2], coarse_vsize, self.F[:,:2])

        self.particle_pos0 = self.F[:,:2]

        self.coarse_vsize = coarse_vsize
        self.fine_vsize = fine_vsize
        self.num_particle = self.particle_pos0.shape[0]
        self.mass_dim = self.C.shape[0]
        self.mass_dist = self.C[:,2]     # coarse mass distribution
        self.obj_name = obj_name



#########################################   TEST   ########################################
if __name__ == '__main__':

    obj_name = object_names[2]

    coarse_vsize = 0.8
    fine_vsize = 0.2

    C = voxelize(poly_coords[obj_name], gt_mass_dist[obj_name], coarse_vsize)
    F = voxelize(poly_coords[obj_name], gt_mass_dist[obj_name], fine_vsize)
    mapping = remap(C[:,:2], coarse_vsize, F[:,:2])

    fig, ax = plt.subplots(1,1)
    polygon = poly_coords[obj_name]
    ax.plot(polygon[:, 0], polygon[:,1], color='deepskyblue')
    ax.plot([polygon[0,0], polygon[-1,0]], 
             [polygon[0,1], polygon[-1,1]], color='deepskyblue')
    ax.axis('equal')

    gt_mass = gt_mass_dist[obj_name]
        
    for i, pt in enumerate(F[:,:2]):
        ax.plot(pt[0], pt[1], c=[C[mapping[i],2]*90/255, 0, 0], marker='o')
    plt.show()