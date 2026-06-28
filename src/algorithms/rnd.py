class RND_SAC(SAC):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.cl_reg_coef = cfg.cl_reg_coef
        self.expert_buffer_size = cfg.expert_buffer_size
        self.replay_buffer_size = cfg.replay_buffer_size
        self.nepochs_offline = cfg.nepochs_offline
        self.env_seq = cfg.env_seq
        self.bc_kl = cfg.bc_kl
        self.distill_kl = cfg.distill_kl
        self.reset_offline_actor = cfg.reset_offline_actor

    