import copy
import random
from functools import partial

from components.episode_buffer import EpisodeBatch
from modules.critics.emaddpg import EnsembleMADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam

from modules.critics.emaddpg import ActivateLayer

dormant_alpha_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
overactive_beta_list = [3.0, 4.0, 5.0]

class EnSetMADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = EnsembleMADDPGCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        # used for reset
        self.initial_mac = copy.deepcopy(mac)
        self.initial_critic = copy.deepcopy(self.critic)
        self.last_reset_network_episode = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.neu_stats = {}
        # Return an iterator over all modules in the network, both the name and the module itself.
        for (name, module) in self.critic.named_modules():
            # if isinstance(module, th.nn.Linear) or isinstance(module, ActivateLayer):
            if isinstance(module, ActivateLayer):
                def hook(module, fea_in, fea_out, module_name):
                    self.neu_stats[module_name] = th.mean(th.abs(fea_out), dim=0)  # (bs, dim) -> (dim)
                module.register_forward_hook(partial(hook, module_name=name))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        for key in self.neu_stats.keys():
            self.neu_stats[key] = th.zeros_like(self.neu_stats[key], device=self.args.device)

        if self.args.random_amplitude_scaling:
            for i in range(batch['obs'].shape[0]):
                z = random.uniform(0.8, 1.2)
                batch['obs'][i] = batch['obs'][i] * z
                batch["state"][i] = batch["state"][i] * z

        if self.args.random_amplitude_scale_and_perturb:  # obs scale, state perturb
            for i in range(batch['obs'].shape[0]):
                z_s = random.uniform(0.8, 1.2)  # default is [0.8, 1.2]
                z_x = random.uniform(-self.args.perturb_alpha, self.args.perturb_alpha)
                z_y = random.uniform(-self.args.perturb_alpha, self.args.perturb_alpha)
                batch['obs'][i] = batch['obs'][i] * z_s  # scale obs
                # agents feature: agent_p_vel: 2, agent p_pos: 2, entity_relative_pos, other_relative_pos, other_vel; agent_states: (ts, n_agents, n_obs) -> (ts, n_agents, n_obs)
                agent_states = batch["state"][i].reshape(-1, self.args.n_agents, self.args.obs_shape)
                agent_x = agent_states[:, :, 2] + z_x
                agent_y = agent_states[:, :, 3] + z_y
                agent_states[:, :, 2] = agent_x
                agent_states[:, :, 3] = agent_y
                batch["state"][i] = agent_states.reshape(-1, self.args.n_agents * self.args.obs_shape)  # perturb agent

        # Train the critic batched
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True,
                                                               critic=self.target_critic, target_mac=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        q_taken = []
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            critic_out, _ = self.critic(inputs, actions[:, t:t+1].detach())  # critic_out: (bs, N)
            critic_out = critic_out.view(batch.batch_size, -1, self.args.mixer_ensemble_N)
            q_taken.append(critic_out)
        q_taken = th.cat(q_taken, dim=1)  # q_taken: (bs, el, N)

        target_vals = []
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            target_critic_out, _ = self.target_critic(target_inputs, target_actions[:, t:t+1].detach())
            target_critic_out = target_critic_out.view(batch.batch_size, -1, self.args.mixer_ensemble_N)
            target_vals.append(target_critic_out)
        target_vals = th.cat(target_vals, dim=1)  # target_vals: (bs, el, N)

        q_taken = q_taken.view(batch.batch_size, -1, self.args.mixer_ensemble_N)
        target_vals = target_vals.view(batch.batch_size, -1, self.args.mixer_ensemble_N)
        targets = rewards.expand_as(target_vals) + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals

        td_error = (q_taken - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            _, _, _ = self.check_neuron_stats(t_env, is_log=True)

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t, select_actions=True)["actions"].view(batch.batch_size,
                                                                                           self.n_agents,
                                                                                           self.n_actions)

            for idx in range(self.n_agents):
                tem_joint_act = actions[:, t:t+1].detach().clone().view(batch.batch_size, -1, self.n_actions)
                tem_joint_act[:, idx] = agent_outs[:, idx]
                q, _ = self.critic(self._build_inputs(batch, t=t), tem_joint_act)
                chosen_action_qvals.append(q.view(batch.batch_size, -1, self.args.mixer_ensemble_N))
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out

        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        # reset network
        if self.args.reset_mode is not None:
            if (episode_num - self.last_reset_network_episode) / self.args.reset_interval >= 1.0:  # RR reset once!
                if self.args.reset_mode == 'shrink_and_perturb':
                    self._shrink_and_perturb_networks(self.args.reset_agent_alpha, self.args.reset_mixer_alpha)
                else:
                    raise NotImplementedError('Unknown reset mode')
                self.last_reset_network_episode = episode_num
                self.logger.console_logger.info('Reset my network')

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q_taken * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
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

        if is_log:
            for i in range(self.args.mixer_ensemble_N):
                for k in range(len(dormant_alpha_list)):
                    self.logger.log_stat("{}th_mixer_{}_dormant".format(i, dormant_alpha_list[k]), mixer_dormant_neuron_count_list[k][i] / mixer_total_neuron_count_list[i], t_env)
                for k in range(len(overactive_beta_list)):
                    self.logger.log_stat("{}th_mixer_{}_overactive".format(i, overactive_beta_list[k]), mixer_overactive_neuron_count_list[k][i] / mixer_total_neuron_count_list[i], t_env)

        return mixer_dormant_neuron_count_list, mixer_overactive_neuron_count_list, mixer_total_neuron_count_list


    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # The centralized critic takes the state input, not observation
        inputs.append(batch["state"][:, t])

        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def _shrink_and_perturb_networks(self, reset_agent_alpha, reset_mixer_alpha):
        for param, initial_param in zip(self.mac.parameters(), self.initial_mac.parameters()):
            param.data.copy_(initial_param.data * (1.0 - reset_agent_alpha) + param.data * reset_agent_alpha)
        for target_param, initial_param in zip(self.target_mac.parameters(), self.initial_mac.parameters()):
            target_param.data.copy_(initial_param.data * (1.0 - reset_agent_alpha) + target_param.data * reset_agent_alpha)

        for param, initial_param in zip(self.critic.parameters(), self.initial_critic.parameters()):
            param.data.copy_(initial_param.data * (1.0 - reset_mixer_alpha) + param.data * reset_mixer_alpha)
        for target_param, initial_param in zip(self.target_critic.parameters(), self.initial_critic.parameters()):
            target_param.data.copy_(initial_param.data * (1.0 - reset_mixer_alpha) + target_param.data * reset_mixer_alpha)
        self.logger.console_logger.info("Shrink and perturb online and target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.initial_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)
        self.initial_critic.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))