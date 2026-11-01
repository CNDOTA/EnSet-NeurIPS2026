import copy
import random
from functools import partial

from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.eqmix import EQMixer
import torch as th
from torch.optim import RMSprop

from modules.mixers.eqmix import ActivateLayer

dormant_alpha_list = [0.1, 0.2, 0.3]
overactive_beta_list = [2.0, 2.5, 3.0]

class EnSetQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.scheme = scheme
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.last_reset_network_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "eqmix":
                self.mixer = EQMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

            # print critic network parameters
            total_mixer_params = sum(p.numel() for p in self.mixer.parameters())
            self.logger.console_logger.info('Centralized critic total parameter number is {}'.format(total_mixer_params))

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        # used for reset
        self.initial_mac = copy.deepcopy(mac)
        self.initial_mixer = copy.deepcopy(self.mixer)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.neu_stats = {}
        # Return an iterator over all modules in the network, both the name and the module itself.
        for (name, module) in self.mixer.named_modules():
            # if isinstance(module, th.nn.Linear) or isinstance(module, ActivateLayer):
            if isinstance(module, ActivateLayer):
                def hook(module, fea_in, fea_out, module_name):
                    self.neu_stats[module_name] = th.mean(th.abs(fea_out), dim=0)  # (bs, dim) -> (dim)
                module.register_forward_hook(partial(hook, module_name=name))

        self.agent_neu_stats = {}
        # Return an iterator over all modules in the network, both the name and the module itself.
        for (name, module) in self.mac.agent.named_modules():
            # if isinstance(module, th.nn.Linear) or isinstance(module, ActivateLayer):
            if isinstance(module, ActivateLayer):
                def hook(module, fea_in, fea_out, module_name):
                    self.agent_neu_stats[module_name] = th.mean(th.abs(fea_out), dim=0)  # (bs, dim) -> (dim)
                module.register_forward_hook(partial(hook, module_name=name))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        for key in self.neu_stats.keys():
            self.neu_stats[key] = th.zeros_like(self.neu_stats[key], device=self.args.device)

        for key in self.agent_neu_stats.keys():
            self.agent_neu_stats[key] = th.zeros_like(self.agent_neu_stats[key], device=self.args.device)

        # batch['obs']: btav
        # batch["state"]: (batch_size, time_step, state_dim)
        # move_feats, enemy_feats, ally_feats, own_feats, last_action, agent_id
        # (visible, distance, relative X, relative Y, health, [shield], [type])
        # state: allies, enemies, [last_action], [timestep]

        if self.args.random_amplitude_scaling:
            for i in range(batch['obs'].shape[0]):
                z = random.uniform(0.8, 1.2)  # default is [0.8, 1.2]
                batch['obs'][i] = batch['obs'][i] * z
                batch["state"][i] = batch["state"][i] * z

        if self.args.random_amplitude_scale_and_perturb:  # obs scale, state perturb
            agent_n_dim = self.args.n_agents * self.args.unit_dim
            enemy_n_dim = self.args.n_enemies * self.args.enemy_dim
            for i in range(batch['obs'].shape[0]):
                z_s = random.uniform(0.8, 1.2)  # default is [0.8, 1.2]
                z_x = random.uniform(-self.args.perturb_alpha, self.args.perturb_alpha)
                z_y = random.uniform(-self.args.perturb_alpha, self.args.perturb_alpha)
                batch['obs'][i] = batch['obs'][i] * z_s  # scale obs
                # agents feature: health, cd/energy, central_x, central_y, others; agent_states: (ts, unit_dim * n_agents) -> (ts, n_agents, unit_dim)
                agent_states = batch["state"][i][:, :agent_n_dim].reshape(-1, self.args.n_agents, self.args.unit_dim)
                agent_x = agent_states[:, :, 2] + z_x
                agent_y = agent_states[:, :, 3] + z_y
                agent_states[:, :, 2] = agent_x
                agent_states[:, :, 3] = agent_y
                batch["state"][i][:, :agent_n_dim] = agent_states.reshape(-1, agent_n_dim)  # perturb agent
                # enemies feature: health, central_x, central_y, others; enemy_states: (ts, enemy_dim * n_enemies) -> (ts, n_enemies, enemy_dim)
                enemy_states = batch["state"][i][:, agent_n_dim: agent_n_dim + enemy_n_dim].reshape(-1, self.args.n_enemies, self.args.enemy_dim)
                enemy_x = enemy_states[:, :, 1] + z_x
                enemy_y = enemy_states[:, :, 2] + z_y
                enemy_states[:, :, 1] = enemy_x
                enemy_states[:, :, 2] = enemy_y
                batch["state"][i][:, agent_n_dim: agent_n_dim + enemy_n_dim] = enemy_states.reshape(-1, enemy_n_dim)  # perturb enemy

        if self.args.random_amplitude_scale_and_agent_permutating:
            for i in range(batch['state'].shape[0]):
                z = random.uniform(0.8, 1.2)  # default is [0.8, 1.2]
                batch['obs'][i] = batch['obs'][i] * z
                agent_permutating_indices = th.randperm(self.args.n_agents)
                agent_states = batch['state'][i][:, :self.args.n_agents * self.args.unit_dim]
                # agent_states: (ts, unit_dim * n_agents) -> (ts, n_agents, unit_dim)
                agent_states = agent_states.reshape(-1, self.args.n_agents, self.args.unit_dim)
                # shuffle agent states
                shuffled_agent_states = agent_states[:, agent_permutating_indices, :].reshape(-1, self.args.n_agents * self.args.unit_dim)
                batch['state'][i][:, :self.args.n_agents * self.args.unit_dim] = shuffled_agent_states


        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            # cur_max_actions: (bs, n_steps, n_agents, 1)
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        # Mix
        if self.mixer is not None:
            # chosen_action_qvals: (bs, el, n_agents) -> (bs, el, n_mixers)
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            _, _, _ = self.check_neuron_stats(t_env, is_log=True)

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:  # RR update once!
            self._update_targets()
            self.last_target_update_episode = episode_num

        if self.args.reset_mode is not None:
            if (episode_num - self.last_reset_network_episode) / self.args.reset_interval >= 1.0:  # RR reset once!
                if self.args.reset_mode == 'shrink_and_perturb':
                    self._shrink_and_perturb_networks(self.args.reset_agent_alpha, self.args.reset_mixer_alpha)
                elif self.args.reset_mode == 'reset':  # fully reset net parameters
                    self._hard_reset_networks()
                else:
                    raise NotImplementedError('Unknown reset mode')
                self.last_reset_network_episode = episode_num
                self.logger.console_logger.info('Reset my network')

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def check_neuron_stats(self, t_env, is_log=False):
        mixer_dormant_neuron_count_list = [[0] * self.args.mixer_ensemble_N for _ in range(len(dormant_alpha_list))]
        mixer_overactive_neuron_count_list = [[0] * self.args.mixer_ensemble_N for _ in range(len(overactive_beta_list))]
        # mixer_normal_neuron_count_list = [0] * self.args.mixer_ensemble_N
        mixer_total_neuron_count_list = [0] * self.args.mixer_ensemble_N

        for name, item in self.neu_stats.items():
            # item: (dim)
            avg = item.mean()

            dormant_count_list = [th.count_nonzero(item <= dormant_alpha * avg).item() for dormant_alpha in dormant_alpha_list]
            overactive_count_list = [th.count_nonzero(item > overactive_beta * avg).item() for overactive_beta in overactive_beta_list]
            # normal_count = th.count_nonzero((dormant_alpha * avg < item) & (item <= overactive_beta * avg)).item()

            # print('{} dead neuron {} in total {} neurons'.format(name, dormant_count_list, item.shape[0]))

            # name like 'hyper_w_final_list.0.0'
            for k in range(len(dormant_alpha_list)):
                mixer_dormant_neuron_count_list[k][int(name.split('.')[1])] += dormant_count_list[k]
            for k in range(len(overactive_beta_list)):
                mixer_overactive_neuron_count_list[k][int(name.split('.')[1])] += overactive_count_list[k]
            mixer_total_neuron_count_list[int(name.split('.')[1])] += item.shape[0]

        # add agent neuron check
        agent_dormant_neuron_count_list = [0 for _ in range(len(dormant_alpha_list))]
        agent_overactive_neuron_count_list = [0 for _ in range(len(overactive_beta_list))]
        agent_total_neuron_count_list = 0

        for name, item in self.agent_neu_stats.items():
            # item: (dim)

            avg = item.mean()

            dormant_count_list = [th.count_nonzero(item <= dormant_alpha * avg).item() for dormant_alpha in dormant_alpha_list]
            overactive_count_list = [th.count_nonzero(item > overactive_beta * avg).item() for overactive_beta in overactive_beta_list]
            # normal_count = th.count_nonzero((dormant_alpha * avg < item) & (item <= overactive_beta * avg)).item()

            # print('{} dead neuron {} in total {} neurons'.format(name, dormant_count_list, item.shape[0]))

            # name like 'A.0'
            for k in range(len(dormant_alpha_list)):
                agent_dormant_neuron_count_list[k] += dormant_count_list[k]
            for k in range(len(overactive_beta_list)):
                agent_overactive_neuron_count_list[k] += overactive_count_list[k]
            agent_total_neuron_count_list += item.shape[0]

        if is_log:
            for i in range(self.args.mixer_ensemble_N):
                for k in range(len(dormant_alpha_list)):
                    self.logger.log_stat("{}th_mixer_{}_dormant".format(i, dormant_alpha_list[k]), mixer_dormant_neuron_count_list[k][i] / mixer_total_neuron_count_list[i], t_env)
                for k in range(len(overactive_beta_list)):
                    self.logger.log_stat("{}th_mixer_{}_overactive".format(i, overactive_beta_list[k]), mixer_overactive_neuron_count_list[k][i] / mixer_total_neuron_count_list[i], t_env)

        for k in range(len(dormant_alpha_list)):
            self.logger.log_stat("agent_{}_dormant".format(dormant_alpha_list[k]), agent_dormant_neuron_count_list[k] / agent_total_neuron_count_list, t_env)
        for k in range(len(overactive_beta_list)):
            self.logger.log_stat("agent_{}_overactive".format(overactive_beta_list[k]), agent_overactive_neuron_count_list[k] / agent_total_neuron_count_list, t_env)


        return mixer_dormant_neuron_count_list, mixer_overactive_neuron_count_list, mixer_total_neuron_count_list


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def _hard_reset_networks(self):
        self.mac.load_state(self.initial_mac)
        self.target_mac.load_state(self.initial_mac)
        if self.mixer is not None:
            self.mixer.load_state_dict(self.initial_mixer.state_dict())
            self.target_mixer.load_state_dict(self.initial_mixer.state_dict())
        self.logger.console_logger.info("Hard reset online and target networks")

    def _shrink_and_perturb_networks(self, reset_agent_alpha, reset_mixer_alpha):
        for param, initial_param in zip(self.mac.parameters(), self.initial_mac.parameters()):
            param.data.copy_(initial_param.data * (1.0 - reset_agent_alpha) + param.data * reset_agent_alpha)
        for target_param, initial_param in zip(self.target_mac.parameters(), self.initial_mac.parameters()):
            target_param.data.copy_(initial_param.data * (1.0 - reset_agent_alpha) + target_param.data * reset_agent_alpha)

        if self.mixer is not None:
            for param, initial_param in zip(self.mixer.parameters(), self.initial_mixer.parameters()):
                param.data.copy_(initial_param.data * (1.0 - reset_mixer_alpha) + param.data * reset_mixer_alpha)
            for target_param, initial_param in zip(self.target_mixer.parameters(), self.initial_mixer.parameters()):
                target_param.data.copy_(initial_param.data * (1.0 - reset_mixer_alpha) + target_param.data * reset_mixer_alpha)
        self.logger.console_logger.info("Shrink and perturb online and target networks")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.initial_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.initial_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
