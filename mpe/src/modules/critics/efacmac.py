import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ActivateLayer(nn.Module):
    def __init__(self, dim, name):
        super(ActivateLayer, self).__init__()
        self.weight = nn.Parameter(th.ones(dim))
        self.weight.requires_grad = False
        self.name = name

    def forward(self, x):
        return x * self.weight


class EnsembleFACMACCritic(nn.Module):
    def __init__(self, scheme, args):
        super(EnsembleFACMACCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1_list = nn.ModuleList([nn.Sequential(nn.Linear(self.input_shape, args.rnn_hidden_dim),
                                                     nn.ReLU(),
                                                     ActivateLayer(args.rnn_hidden_dim, 'fc1'))
                                       for _ in range(self.args.mixer_ensemble_N)])
        self.fc2_list = nn.ModuleList([nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
                                                     nn.ReLU(),
                                                     ActivateLayer(args.rnn_hidden_dim, 'fc2'))
                                       for _ in range(self.args.mixer_ensemble_N)])
        self.fc3_list = nn.ModuleList([nn.Linear(args.rnn_hidden_dim, 1) for _ in range(self.args.mixer_ensemble_N)])

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        bs = inputs.size(0)
        if actions is not None:
            inputs = th.cat([inputs.view(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        q_tot_list = []
        for i in range(self.args.mixer_ensemble_N):
            x = self.fc1_list[i](inputs)
            x = self.fc2_list[i](x)
            q = self.fc3_list[i](x)
            q = q.view(bs, 1)  # (bs, 1)
            q_tot_list.append(q)
        q_tot_list = th.cat(q_tot_list, dim=1)  # (bs, N)
        return q_tot_list, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape


class FACMACDiscreteCritic(nn.Module):
    def __init__(self, scheme, args):
        super(FACMACDiscreteCritic, self).__init__()
        self.args = args
        self.n_actions = scheme["actions_onehot"]["vshape"][0]
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme) + self.n_actions
        self.output_type = "q"
        self.hidden_states = None

        # Set up network layers
        self.fc1 = nn.Linear(self.input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        self.hidden_states = None

    def forward(self, inputs, actions, hidden_state=None):
        if actions is not None:
            inputs = th.cat([inputs.reshape(-1, self.input_shape - self.n_actions),
                             actions.contiguous().view(-1, self.n_actions)], dim=-1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, hidden_state

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        return input_shape