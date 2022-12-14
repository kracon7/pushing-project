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

class BacktrackingMomentum:
    def __init__(self, parameters, bounds, body_poses_gt, momentum=0.7, alpha=0.1, beta=0.8, lr_0=0.01, lr_min=1e-6):
        '''
        Momentum optimizer with backtracking line search for step size
        Args:
            parameters -- Dictionary, {param_name: ndarray}
            bounds -- Dictionary, {param_name: [lower_bound, upper_bound]}
            momentum -- hyperparameter
        '''
        self.parameters = parameters
        self.bounds = bounds
        self.momentum = momentum
        self.momentum_buffer = {k: np.zeros_like(v) for (k, v) in parameters.items()}
        self.body_poses_gt = body_poses_gt
        self.alpha = alpha
        self.beta = beta
        self.lr_0 = lr_0
        self.lr_min = lr_min

    def step(self, sim, grads):
        grads_new = {k: self.momentum_buffer[k] * self.momentum + v * (1 - self.momentum) 
                for (k, v) in grads.items()}
        self.momentum_buffer = grads_new
        
        # line search for step size
        lr = self.line_search(sim, grads_new)

        # update parameters
        self.parameters = {k: (v - lr * grads_new[k]).clip(*self.bounds[k])
                        for (k, v) in self.parameters.items()}

        return self.parameters.copy()

    def line_search(self, sim, grads):
        lr = self.lr_0              # Initial learning rate
        loss_prev = 1e9         
        sim_steps = max(sim.loss_steps) + 1

        params = {k: sim.get_parameter(k) for (k, v) in grads.items()}

        # print("Line search direction: %.5f, %.5f"%(grads["mass"][0], grads["friction"][0]))

        while lr >= self.lr_min:
            params_new = {k: v - lr * grads[k] for (k, v) in params.items()}
            if not self.check_bounds(params_new):
                # print("Parameters out of bounds after update, gradient step is too large!")
                lr *= self.beta
                continue
            
            # Update sim parameters
            for k, v in params_new.items():
                sim.update_parameter(v, k)

            sim.loss_backtrack[None] = 0
            sim.run(sim_steps)
            sim.compute_loss_backtrack(self.body_poses_gt)

            if sim.loss[None] > sim.loss_backtrack[None] and \
                                    sim.loss_backtrack[None] > loss_prev:
                break
            else:
                lr *= self.beta

            loss_prev = sim.loss_backtrack[None]
            # print("Backtracking search at learning rate %.7f \
            #         loss: %.9f, loss_backtracking: %.9f"%\
            #         (lr, sim.loss[None], sim.loss_backtrack[None]))
        
        return lr

    def check_bounds(self, parameters):
        for k, v in parameters.items():
            if (v < self.bounds[k][0]).any() or (v > self.bounds[k][1]).any():
                return False
            else:
                return True


class Backtracking:
    def __init__(self, body_poses_gt, param_name, alpha=0.1, beta=0.8, lr_0=0.01, lr_min=1e-6):
        '''
        Backtracking line search for step size
        Args:
            body_poses_gt -- ndarray (batch, timestep, 3, ).
                             Ground truth body qpos and body rpos
            param_type -- string, 
                          "mass" or "friction"
            grad -- gradient of the parameter
            sim_steps -- number of simulation steps
            loss_steps -- list of the timesteps added to loss computation
        Return:
            lr -- optimal step size
        '''
        self.body_poses_gt = body_poses_gt
        self.param_name = param_name
        self.alpha = alpha
        self.beta = beta
        self.lr_0 = lr_0
        self.lr_min = lr_min

    def line_search(self, sim, grad, sim_steps, loss_steps):
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
            sim.compute_loss_backtrack(self.body_poses_gt)

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