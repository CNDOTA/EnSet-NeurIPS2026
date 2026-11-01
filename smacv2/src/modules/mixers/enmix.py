import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActivateLayer(nn.Module):
    def __init__(self, dim, name):
        super(ActivateLayer, self).__init__()
        self.weight = nn.Parameter(th.ones(dim))
        self.weight.requires_grad = False
        self.name = name

    def forward(self, x):
        return x * self.weight


class EMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(EMixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape)) 
        
        # hyper w1 b1
        self.hyper_w1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                                          nn.ReLU(inplace=True),
                                                          ActivateLayer(args.hypernet_embed, 'w_1_1'),
                                                          nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim),
                                                          )
                                            for _ in range(self.args.mixer_ensemble_N)])
        self.hyper_b1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, self.embed_dim)) for _ in range(self.args.mixer_ensemble_N)])
        
        # hyper w2 b2
        self.hyper_w2_list = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                                          nn.ReLU(inplace=True),
                                                          ActivateLayer(args.hypernet_embed, 'w_2_1'),
                                                          nn.Linear(args.hypernet_embed, self.embed_dim))
                                            for _ in range(self.args.mixer_ensemble_N)])
        self.hyper_b2_list = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                                     nn.ReLU(inplace=True),
                                                     ActivateLayer(self.embed_dim, 'b_2_1'),
                                                     nn.Linear(self.embed_dim, 1))
                                       for _ in range(self.args.mixer_ensemble_N)])

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()
        
        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        q_tot_list = []
        for i in range(self.args.mixer_ensemble_N):
            # First layer
            w1 = self.hyper_w1_list[i](states).view(-1, self.n_agents, self.embed_dim) # b * t, n_agents, emb
            b1 = self.hyper_b1_list[i](states).view(-1, 1, self.embed_dim)

            # Second layer
            w2 = self.hyper_w2_list[i](states).view(-1, self.embed_dim, 1) # b * t, emb, 1
            b2= self.hyper_b2_list[i](states).view(-1, 1, 1)

            if self.abs:
                w1 = w1.abs()
                w2 = w2.abs()

            # Forward
            hidden = F.elu(th.matmul(qvals, w1) + b1) # b * t, 1, emb
            y = th.matmul(hidden, w2) + b2 # b * t, 1, 1
            q_tot_list.append(y.view(b, t, 1))
        q_tot_list = th.cat(q_tot_list, dim=2)  # (bs, el, N)
        return q_tot_list  # b,t,N
    
