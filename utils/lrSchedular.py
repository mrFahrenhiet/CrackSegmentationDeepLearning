class OneCycleLR:
    def __init__(self,
                 optimizer,
                 num_steps,
                 lr_range=(0.1, 1.),
                 momentum_range=(0.85, 0.95),
                 annihilation_frac=0.1,
                 reduce_factor=0.01,
                 last_step=-1):

        self.optimizer = optimizer

        self.num_steps = num_steps

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum

        self.num_cycle_steps = int(num_steps * (1. - annihilation_frac))
        self.final_lr = self.min_lr * reduce_factor

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    def step(self):
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.num_cycle_steps // 2:
            scale = current_step / (self.num_cycle_steps // 2)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            scale = (current_step - self.num_cycle_steps // 2) / (self.num_cycle_steps - self.num_cycle_steps // 2)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_steps:
            scale = (current_step - self.num_cycle_steps) / (self.num_steps - self.num_cycle_steps)
            lr = self.min_lr - (self.min_lr - self.final_lr) * scale
            momentum = None
        else:
            return

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum
