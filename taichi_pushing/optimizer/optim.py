import numpy as np

class Optimizer:
    def __init__(self, parameters: np.ndarray, **kwargs):
        self.kwargs = kwargs
        self.lr = kwargs['lr']
        self.bounds = kwargs['bounds']
        self.parameters = parameters
        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def _step(self, grads):
        raise NotImplementedError

    def step(self, grads):
        assert grads.shape == self.parameters.shape
        self.parameters[:] = self._step(grads).clip(*self.bounds)
        return self.parameters.copy()


class Momentum(Optimizer):
    def initialize(self):
        self.momentum_buffer = np.zeros_like(self.parameters).astype(np.float64)
        self.momentum = self.kwargs['momentum']

    def _step(self, grads):
        grads = self.momentum_buffer * self.momentum + grads * (1 - self.momentum)
        self.momentum_buffer[:] = grads
        return self.parameters[:] - self.lr * grads

    @classmethod
    def default_config(cls):
        cfg = Optimizer.default_config()
        cfg.momentum = 0.9
        return cfg


class Adam(Optimizer):
    def initialize(self):
        self.momentum_buffer = np.zeros_like(self.parameters).astype(np.float64)
        self.v_buffer = np.zeros_like(self.momentum_buffer).astype(np.float64)
        self.iter = 0

    def _step(self, grads):
        gd = grads.reshape(*self.parameters.shape)
        beta_1 = self.cfg.beta_1
        beta_2 = self.cfg.beta_2
        epsilon = self.cfg.epsilon
        m_t = beta_1 * self.momentum_buffer + (1 - beta_1) * gd  # updates the moving averages of the gradient
        v_t = beta_2 * self.v_buffer + (1 - beta_2) * (gd * gd)  # updates the moving averages of the squared gradient
        self.momentum_buffer[:] = m_t
        self.v_buffer[:] = v_t

        m_cap = m_t / (1 - (beta_1 ** (self.iter + 1)))  # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta_2 ** (self.iter + 1)))  # calculates the bias-corrected estimates

        self.iter += 1
        return self.parameters - (self.lr * m_cap) / (np.sqrt(v_cap) + epsilon)

    @classmethod
    def default_config(cls):
        cfg = Optimizer.default_config()
        cfg.beta_1 = 0.9
        cfg.beta_2 = 0.999
        cfg.epsilon = 1e-8
        return cfg

class Backtracking:
    def __init__(self, gt_trajectory, param_name, alpha=0.1, beta=0.8, lr_0=0.01, lr_min=1e-4):
        '''
        Backtracking line search for step size
        Args:
            gt_trajectory -- ndarray (batch, timestep, 3, ).
                             Ground truth body qpos and body rpos
            param_type -- string, 
                          "mass" or "friction"
            grad -- gradient of the parameter
            sim_steps -- number of simulation steps
            loss_steps -- list of the timesteps added to loss computation
        Return:
            lr -- optimal step size
        '''
        self.gt_trajectory = gt_trajectory
        self.param_name = param_name
        self.alpha = alpha
        self.beta = beta
        self.lr_0 = lr_0
        self.lr_min = lr_min

    def backtracking(self, sim, grad, sim_steps, loss_steps):
        '''
        Backtracking line search for step size
        Args:
            sim -- Taichi simulator
            param_type -- string, 
                          "mass" or "friction"
            grad -- gradient of the parameter
            sim_steps -- number of simulation steps
            loss_steps -- list of the timesteps added to loss computation
        Return:
            lr -- optimal step size
        '''
        
        lr = self.lr_0              # Initial learning rate
        loss_prev = 1e9         

        param = sim.get_parameter(self.param_name)

        while lr >= self.lr_min:
            param_new = param - lr * grad
            if (param_new <= 1e-2).any():
                print("Negative update after update, gradient step is too large!")
                lr *= self.beta
                continue
            
            sim.update_parameter(param_new, self.param_name)

            sim.loss_backtrack[None] = 0
            sim.run(sim_steps)

            for b in sim.u.keys():
                for idx in loss_steps:
                    sim.compute_loss_backtrack(b, idx, 
                                    self.gt_trajectory[b, idx][0],
                                    self.gt_trajectory[b, idx][1], 
                                    self.gt_trajectory[b, idx][2])

            if sim.loss[None] > sim.loss_backtrack[None] and \
                                    sim.loss_backtrack[None] > loss_prev:
                break
            else:
                lr *= self.beta

            loss_prev = sim.loss_backtrack[None]
            print("Backtracking search at learning rate %.7f \
                    loss: %.9f, loss_backtracking: %.9f"%\
                    (lr, sim.loss[None], sim.loss_backtrack[None]))
        
        return lr