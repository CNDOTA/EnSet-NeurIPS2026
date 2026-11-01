import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
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

        self.log_stats_t = -self.args.learner_log_interval - 1

        # for churn on refer batch
        self.ref_mac = copy.deepcopy(mac)
        self.ref_mixer = copy.deepcopy(self.mixer)


    def infer_refer_batch_action_values(self, ref_batch: EpisodeBatch, ref_mac, ref_mixer):
        # Get the relevant quantities
        actions = ref_batch["actions"][:, :-1]
        avail_actions = ref_batch["avail_actions"]
        # Calculate estimated Q-Values
        mac_out = []
        ref_mac.init_hidden(ref_batch.batch_size)
        for t in range(ref_batch.max_seq_length):
            agent_outs = ref_mac.forward(ref_batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions_ref_batch = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # current net, last batch
        # Mix
        if ref_mixer is not None:
            chosen_action_qvals_ref_batch = ref_mixer(chosen_action_qvals, ref_batch["state"][:, :-1])
        # mac_out is individual Q-values
        return mac_out, cur_max_actions_ref_batch, chosen_action_qvals_ref_batch


    def train(self, batch: EpisodeBatch, ref_batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
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

        # infer the ref Q and ref action on refer batch with old net
        with th.no_grad():
            last_ind_qvals_ref_batch, last_max_actions_ref_batch, last_mixer_qvals_ref_batch = self.infer_refer_batch_action_values(ref_batch, self.ref_mac, self.ref_mixer)
        # infer the ref Q and ref action on refer batch with cur net
        cur_ind_qvals_ref_batch, cur_max_actions_ref_batch, cur_mixer_qvals_ref_batch = self.infer_refer_batch_action_values(ref_batch, self.mac, self.mixer)

        # policy churn has no gradient as actions are discrete
        policy_churn = (last_max_actions_ref_batch != cur_max_actions_ref_batch).sum() / th.ones_like(cur_max_actions_ref_batch).sum()
        ma_action_equal = (last_max_actions_ref_batch == cur_max_actions_ref_batch).float().mean(dim=2)
        ma_policy_churn = (ma_action_equal < 1).sum() / th.ones_like(ma_action_equal).sum()

        ref_terminated = ref_batch["terminated"][:, :-1].float()
        ref_mask = ref_batch["filled"][:, :-1].float()
        ref_mask[:, 1:] = ref_mask[:, 1:] * (1 - ref_terminated[:, :-1])
        # ref_mask: (bs, n_steps - 1, 1)

        # for qmix, calculate individual value churn and ma value churn
        # ind_value_churn: (bs, n_steps, n_agents, n_actions) -> (bs, n_steps, n_agents) -> (bs, n_steps, 1)
        ind_value_churn = th.abs(cur_ind_qvals_ref_batch - last_ind_qvals_ref_batch).mean(dim=3).mean(dim=2, keepdim=True)
        ind_value_churn = (ind_value_churn[:, :-1] * ref_mask).sum() / ref_mask.sum()
        # ma_value_churn: (bs, n_steps - 1, 1)
        ma_value_churn = th.abs(cur_mixer_qvals_ref_batch - last_mixer_qvals_ref_batch)
        ma_value_churn = (ma_value_churn * ref_mask).sum() / ref_mask.sum()

        # # ind_value_churn = th.abs((cur_ind_qvals_ref_batch - last_ind_qvals_ref_batch)).mean()
        # # ma_value_churn = th.abs((cur_mixer_qvals_ref_batch - last_mixer_qvals_ref_batch)).mean()

        # churn loss
        ind_value_churn_loss = self.args.ind_value_churn_lambda * ind_value_churn
        ma_value_churn_loss = self.args.ma_value_churn_lambda * ma_value_churn
        loss += ind_value_churn_loss + ma_value_churn_loss

        # before update, save mac and mixer to last mac and last mixer
        self.ref_mac.load_state(self.mac)
        self.ref_mixer.load_state_dict(self.mixer.state_dict())

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # log info in current batch and last batch
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            self.logger.log_stat("policy_churn", policy_churn.item(), t_env)
            self.logger.log_stat("ma_policy_churn", ma_policy_churn.item(), t_env)
            self.logger.log_stat("ind_value_churn", ind_value_churn.item(), t_env)
            self.logger.log_stat("ma_value_churn", ma_value_churn.item(), t_env)

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.ref_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.ref_mixer.cuda()

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