import os
import math
import taichi as ti

class Defaults:
    """Aggregates general simulation parameters defaults.
    """
    # Dimensions
    DIM = 2

    FPS = 1000
    DT = 1.0 / FPS
    DTYPE = ti.f64
    DEVICE = ti.cpu

    def __init__(self):
        pass

class Recorder:
    """Records simulations into a series of image frames.
    """
    def __init__(self, dt, screen, path=os.path.join('videos', 'frames')):
        self.dt = dt
        self.prev_t = 0.
        self.frame = 0
        self.screen = screen
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    def record(self, t):
        if t - self.prev_t >= self.dt:
            pygame.image.save(self.screen,
                              os.path.join(self.path,
                                           '%07d.bmp'%(self.frame)))
            self.frame += 1
            self.prev_t += self.dt


def plot(y_axis, x_axis=None):
    import matplotlib.pyplot as plt
    if x_axis is None:
        x_axis = range(len(y_axis))
    else:
        x_axis = [x.item() if x.__class__ is torch.Tensor else x for x in x_axis]
    y_axis = [y.item() if y.__class__ is torch.Tensor else y for y in y_axis]
    plt.plot(x_axis, y_axis)
    plt.show()

def rel_pose(p1, p2):
    ''' 
    Compute the relative translation and rotation between 2 array of
    particle positions of composite objects.
    The object origin is assumed at the first particle center
    Input:
        p1, p2 -- torch tensors of shape (N, 2)
    Output:
        trans -- translation from p1 origin to p2 origin
        theta -- rotation from p1 to p2
    '''
    trans = p2[0] - p1[0]

    p1 = p1[1:] - p1[0]
    p2 = p2[1:] - p2[0]
    s = (p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]) / \
        (torch.norm(p1, dim=-1) * torch.norm(p2, dim=-1))
    c = torch.bmm(p1.unsqueeze(1), p2.unsqueeze(-1)).reshape(-1) / \
        (torch.norm(p1, dim=-1) * torch.norm(p2, dim=-1))
    theta = torch.mean(torch.atan2(s, c), dim=0, keepdim=True)

    return torch.cat([theta, trans])