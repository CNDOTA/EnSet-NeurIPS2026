import copy

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


class EQMixer(nn.Module):
    def __init__(self, args):
        super(EQMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_enemies = args.n_enemies
        self.agent_dim = args.unit_dim
        self.enemy_dim = args.enemy_dim
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim * self.n_agents) for _ in range(self.args.mixer_ensemble_N)])
            self.hyper_w_final_list = nn.ModuleList([nn.Linear(self.state_dim, self.embed_dim) for _ in range(self.args.mixer_ensemble_N)])
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                               nn.ReLU(),
                                                               ActivateLayer(hypernet_embed, 'w_1_1'),
                                                               nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
                                                               )
                                                 for _ in range(self.args.mixer_ensemble_N)])
            self.hyper_w_final_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                                   nn.ReLU(),
                                                                   ActivateLayer(hypernet_embed, 'w_f_1'),
                                                                   nn.Linear(hypernet_embed, self.embed_dim),
                                                                   )
                                                     for _ in range(self.args.mixer_ensemble_N)])
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, self.embed_dim))
                                             for _ in range(self.args.mixer_ensemble_N)])

        # V(s) instead of a bias for the last layers
        self.V_list = nn.ModuleList([nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                                   nn.ReLU(),
                                                   ActivateLayer(self.embed_dim, 'V_1'),
                                                   nn.Linear(self.embed_dim, 1),
                                                   )
                                     for _ in range(self.args.mixer_ensemble_N)])




    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        el = agent_qs.size(1)

        states = states.reshape(-1, self.state_dim)  # (bs * el, state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # (bs * el, 1, n_agents)

        # shuffled_states = copy.deepcopy(states)
        # if self.args.random_agent_permutating:
            # agent_shuffle_indices = th.randperm(self.n_agents)
            # # agent feature: health, cd/energy, central_x, central_y, others
            # agent_states = shuffled_states[:, :self.agent_dim * self.n_agents]  # get agent own features from state
            # agent_states = agent_states.reshape(-1, self.n_agents, self.agent_dim)
            # shuffled_agent_states = agent_states[:, agent_shuffle_indices, :].reshape(-1, self.agent_dim * self.n_agents)
            # # update
            # shuffled_states[:, :self.agent_dim * self.n_agents] = shuffled_agent_states

        q_tot_list = []
        for i in range(self.args.mixer_ensemble_N):
            # First layer
            w1 = th.abs(self.hyper_w_1_list[i](states))
            b1 = self.hyper_b_1_list[i](states)
            w1 = w1.view(-1, self.n_agents, self.embed_dim)
            b1 = b1.view(-1, 1, self.embed_dim)
            # agent_qs: (bs*el, 1, n_agents), w1: (bs*el, n_agents, embed_dim)
            hidden = F.elu(th.bmm(agent_qs, w1) + b1)  # (bs*el, 1, emb)
            # Second layer
            w_final = th.abs(self.hyper_w_final_list[i](states))
            w_final = w_final.view(-1, self.embed_dim, 1)  # (bs, emb, 1)
            # State-dependent bias
            v = self.V_list[i](states).view(-1, 1, 1)  # (bs*el, 1, 1)
            # Compute final output
            y = th.bmm(hidden, w_final) + v  # (bs*el, 1, 1)
            # Reshape and return
            q_tot = y.view(bs, -1, 1)  # (bs, el, 1)
            q_tot_list.append(q_tot)
        q_tot_list = th.cat(q_tot_list, dim=2)  # (bs, el, N)
        return q_tot_list





