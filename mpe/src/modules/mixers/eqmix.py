import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.critics.efacmac import ActivateLayer

class EnsembleQMixer(nn.Module):
    def __init__(self, args):
        super(EnsembleQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.q_embed_dim = getattr(self.args, "q_embed_dim", 1)

        self.hyper_w_1_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim * self.n_agents * self.q_embed_dim) for _ in range(self.args.mixer_ensemble_N)])
        self.hyper_w_final_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.args.mixer_ensemble_N)])

        if getattr(self.args, "hypernet_layers", 1) > 1:
            raise NotImplementedError('not implemented')

        # Initialise the hyper networks with a fixed variance, if specified
        if self.args.hyper_initialization_nonzeros > 0:
            std = self.args.hyper_initialization_nonzeros ** -0.5
            self.hyper_w_1.weight.data.normal_(std=std)
            self.hyper_w_1.bias.data.normal_(std=std)
            self.hyper_w_final.weight.data.normal_(std=std)
            self.hyper_w_final.bias.data.normal_(std=std)

        # State dependent bias for hidden layer
        self.hyper_b_1_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.args.mixer_ensemble_N)])

        # V(s) instead of a bias for the last layers
        self.V_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                                   nn.ReLU(),
                                                   ActivateLayer(self.embed_dim, 'V_1'),
                                                   nn.Linear(self.embed_dim, 1))
                                     for _ in range(self.args.mixer_ensemble_N)])

        if self.args.gated:
            self.gate = nn.Parameter(th.ones(size=(1,)) * 0.5)

    def forward(self, agent_qs_list, states):
        # agent_qs_list: (bs, n_agents, N)
        bs = agent_qs_list.size(0)
        states = states.reshape(-1, self.state_dim)

        q_tot_list = []
        for i in range(self.args.mixer_ensemble_N):
            agent_qs = agent_qs_list[:, :, i].view(-1, 1, self.n_agents * self.q_embed_dim)  # (bs, 1, n_agents)
            # First layer
            w1 = th.abs(self.hyper_w_1_list[i](states))
            b1 = self.hyper_b_1_list[i](states)
            w1 = w1.view(-1, self.n_agents*self.q_embed_dim, self.embed_dim)
            b1 = b1.view(-1, 1, self.embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)
            # Second layer
            w_final = th.abs(self.hyper_w_final_list[i](states))
            w_final = w_final.view(-1, self.embed_dim, 1)
            # State-dependent bias
            v = self.V_list[i](states).view(-1, 1, 1)
            # Skip connections
            s = 0
            if self.args.skip_connections:
                s = agent_qs.sum(dim=2, keepdim=True)

            if self.args.gated:
                y = th.bmm(hidden, w_final) * self.gate + v + s
            else:
                # Compute final output
                y = th.bmm(hidden, w_final) + v + s
            # Reshape and return
            q_tot = y.view(bs, -1, 1)
            q_tot_list.append(q_tot)
        q_tot_list = th.cat(q_tot_list, dim=2)  # (bs, 1, N)
        return q_tot_list