"""
PyTorch port of deep_adfq/deepadfq.py, closely following the original structure and logic.
Includes all options, logging, and ADFQ update logic (using numpy for posterior_adfq/v2 for now).
"""
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import zipfile
import cloudpickle
import random
from collections import deque
from scipy.special import logsumexp

from . import models, ActWrapper, DQNAgent
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from .utils import linear_schedule

REW_VAR_0 = 1e-3
DTYPE = torch.float32


def posterior_adfq(n_means, n_vars, c_mean, c_var, reward, discount, terminal,
                   varTH=1e-20, REW_VAR=REW_VAR_0, scale_factor=1.0, asymptotic=False,
                   asymptotic_trigger=1e-20, noise=0.0, noise_c=0.0, batch=False):
    """PyTorch version of posterior_adfq"""
    device = n_means.device
    noise = noise / scale_factor
    noise_c = noise_c / scale_factor
    c_var = c_var + noise_c
    target_vars = discount * discount * n_vars
    t = asymptotic_trigger / scale_factor

    if batch:
        batch_size = n_means.shape[0]
        target_means = reward.unsqueeze(1) + discount * n_means
        stats = posterior_adfq_batch_helper(target_means, target_vars, c_mean,
                                            c_var, discount, scale_factor=scale_factor, asymptotic=asymptotic,
                                            noise=noise)
        reward = reward.to(DTYPE)
        terminal = terminal.to(torch.int)
        if asymptotic:
            is_asymptotic = torch.prod((n_vars <= t), dim=-1) * torch.prod((c_var <= t), dim=-1)
    else:
        target_means = reward + discount * n_means
        sorted_idx = torch.flip(torch.argsort(target_means), dims=[-1])
        stats = posterior_adfq_helper(target_means, target_vars, c_mean, c_var,
                                      discount, scale_factor=scale_factor, sorted_idx=sorted_idx,
                                      asymptotic=asymptotic, noise=noise)
        stats = stats.unsqueeze(0)
        if asymptotic and (n_vars <= t).all() and (c_var <= t):
            b_rep = torch.argmin(stats[:, :, 2], dim=-1)
            weights = torch.zeros(len(stats), device=device)
            weights[b_rep] = 1.0
            return stats[b_rep, 0], torch.maximum(stats[b_rep, 1], torch.tensor(varTH, device=device)), (
                stats[:, :, 0], stats[:, :, 1], weights)

    logk = stats[:, :, 2] - torch.max(stats[:, :, 2], dim=-1, keepdim=batch)[0]
    weights = torch.exp(logk - torch.logsumexp(logk, dim=-1, keepdim=batch)).to(DTYPE)
    v = weights * stats[:, :, 0]
    mean_new = torch.sum(v, dim=-1, keepdim=batch)
    var_new = torch.sum(weights * stats[:, :, 1], dim=-1) \
              + torch.sum(v * (stats[:, :, 0] - mean_new), dim=-1) / scale_factor

    var_new = (1. - terminal) * var_new + terminal * 1. / (1. / c_var + scale_factor / (REW_VAR + noise))
    mean_new = (1. - terminal) * mean_new.squeeze() + terminal * var_new * (
            c_mean / c_var + scale_factor * reward / (REW_VAR + noise))

    if torch.isnan(mean_new).any() or torch.isinf(mean_new).any():
        print("NaN or Inf value at Mean")
        import pdb;
        pdb.set_trace()
    if torch.isnan(var_new).any() or torch.isinf(var_new).any():
        print("NaN or Inf value at Var")
        import pdb;
        pdb.set_trace()

    if batch:
        return mean_new.squeeze(), torch.maximum(torch.tensor(varTH, device=device), var_new.squeeze()), (
            stats[:, :, 0].squeeze(), stats[:, :, 1].squeeze(), weights.squeeze())
    else:
        return mean_new[0], torch.maximum(torch.tensor(varTH, device=device), var_new[0]), (
            stats[:, :, 0].squeeze(), stats[:, :, 1].squeeze(), weights.squeeze())


def posterior_adfq_batch_helper(target_means, target_vars, c_mean, c_var,
                                discount, scale_factor=1.0, asymptotic=False, noise=0.0):
    """PyTorch version of posterior_adfq_batch_helper"""
    batch_stats = []
    sorted_idx = torch.flip(torch.argsort(target_means), dims=[-1])
    for k in range(target_means.shape[0]):
        noise_k = noise[k] if isinstance(noise, torch.Tensor) else noise
        stats = posterior_adfq_helper(target_means[k], target_vars[k], c_mean[k],
                                      c_var[k], discount, sorted_idx[k], scale_factor, asymptotic,
                                      noise=noise_k)
        batch_stats.append(stats)
    return torch.stack(batch_stats)


def posterior_adfq_helper(target_means, target_vars, c_mean, c_var, discount,
                          sorted_idx, scale_factor=1.0, asymptotic=False, noise=0.0):
    """PyTorch version of posterior_adfq_helper"""
    device = target_means.device
    anum = target_means.shape[-1]
    dis2 = discount * discount
    bar_vars = 1. / (1. / c_var + 1. / (target_vars + noise))
    bar_means = bar_vars * (c_mean / c_var + target_means / (target_vars + noise))
    add_vars = c_var + target_vars + noise
    stats = torch.zeros((anum, 3), device=device)

    for (j, b) in enumerate(sorted_idx):
        b_primes = [c for c in sorted_idx if c != b]
        outcome = iter_search(anum, b_primes, target_means, target_vars,
                              target_means[b], bar_means[b], bar_vars[b], add_vars[b], c_mean,
                              discount, asymptotic=asymptotic, scale_factor=scale_factor)
        if outcome is not None:
            stats[b] = torch.tensor(outcome)
        else:
            print("Could not find a matching mu star")
    return torch.stack([stats[i] for i in range(anum)]).to(DTYPE)


def iter_search(anum, b_primes, target_means, target_vars, target_means_b,
                bar_means_b, bar_vars_b, add_vars_b, c_mean, discount, asymptotic=False,
                scale_factor=1.0):
    """PyTorch version of iter_search"""
    device = target_means.device
    upper = 1e+20
    dis2 = discount * discount

    for i in range(anum):
        n_target_means = torch.stack([target_means[b_primes[k]] for k in range(i)]) if i > 0 else torch.tensor([],
                                                                                                               device=device)
        n_target_vars = torch.stack([target_vars[b_primes[k]] for k in range(i)]) if i > 0 else torch.tensor([],
                                                                                                             device=device)

        if i == (anum - 1):
            lower = -1e+20
        else:
            lower = target_means[b_primes[i]]

        if i == 0:
            mu_star = bar_means_b
        else:
            mu_star = (torch.sum(n_target_means / n_target_vars) + bar_means_b / bar_vars_b) \
                      / (1. / bar_vars_b + torch.sum(1. / n_target_vars))

        if (mu_star >= lower) and (mu_star <= upper):
            if i == 0:
                var_star = bar_vars_b
            else:
                var_star = 1. / (1. / bar_vars_b + torch.sum(1. / n_target_vars))

            if asymptotic:
                if i == 0:
                    logk = (target_means_b - c_mean) ** 2 / add_vars_b + (mu_star - bar_means_b) ** 2 / bar_vars_b
                else:
                    logk = (target_means_b - c_mean) ** 2 / add_vars_b \
                           + (mu_star - bar_means_b) ** 2 / bar_vars_b \
                           + torch.sum((n_target_means - mu_star) ** 2 / n_target_vars)
            else:
                if i == 0:
                    logk = 0.5 * (torch.log(var_star) - torch.log(bar_vars_b) \
                                  - torch.log(torch.tensor(2.0) * torch.pi) - torch.log(add_vars_b)) \
                           - 0.5 / scale_factor * ((target_means_b - c_mean) ** 2 \
                                                   / add_vars_b + (mu_star - bar_means_b) ** 2 / bar_vars_b)
                else:
                    logk = 0.5 * (torch.log(var_star) - torch.log(bar_vars_b) \
                                  - torch.log(torch.tensor(2.0) * torch.pi) - torch.log(add_vars_b)) \
                           - 0.5 / scale_factor * ((target_means_b - c_mean) ** 2 \
                                                   / add_vars_b + (mu_star - bar_means_b) ** 2 / bar_vars_b \
                                                   + torch.sum((n_target_means - mu_star) ** 2 / n_target_vars))
            return mu_star, var_star, logk
        else:
            upper = lower


def posterior_adfq_v2(n_means, n_vars, c_mean, c_var, reward, discount,
                      terminal, varTH=1e-20, REW_VAR=REW_VAR_0, logEps=-1e+20,
                      scale_factor=1.0, asymptotic=False, asymptotic_trigger=1e-20,
                      batch=False, noise=0.0):
    """PyTorch version of posterior_adfq_v2"""
    device = n_means.device

    if batch:
        batch_size = len(n_means)
        c_mean = c_mean.unsqueeze(1)
        c_var = c_var.unsqueeze(1)
        reward = reward.unsqueeze(1)
        terminal = terminal.unsqueeze(1)

    target_means = reward + discount * n_means
    target_vars = discount * discount * n_vars
    bar_vars = 1. / (1. / c_var + 1. / (target_vars + noise))
    bar_means = bar_vars * (c_mean / c_var + target_means / (target_vars + noise))
    add_vars = c_var + target_vars + noise

    sorted_idx = torch.argsort(target_means, dim=int(batch))

    if batch:
        ids = torch.arange(0, batch_size, device=device)
        bar_targets = target_means[ids, sorted_idx[:, -1], None] * torch.ones_like(target_means)
        bar_targets[ids, sorted_idx[:, -1]] = target_means[ids, sorted_idx[:, -2]]
    else:
        bar_targets = target_means[sorted_idx[-1]] * torch.ones_like(target_means)
        bar_targets[sorted_idx[-1]] = target_means[sorted_idx[-2]]

    thetas = torch.heaviside(bar_targets - bar_means, torch.tensor(0.0, device=device))
    t = asymptotic_trigger / scale_factor

    if asymptotic:
        if (n_vars <= t).all() and (c_var <= t):
            b_rep = torch.argmin((target_means - c_mean) ** 2 - 2 * add_vars * logEps * thetas)
            weights = torch.zeros_like(target_means)
            weights[b_rep] = 1.0
            return bar_means[b_rep], torch.maximum(torch.tensor(varTH, device=device), bar_vars[b_rep]), (
                bar_means, bar_vars, weights)

    log_weights = -0.5 * (torch.log(2 * torch.pi) + torch.log(add_vars) + (
            c_mean - target_means) ** 2 / add_vars / scale_factor) + logEps * thetas
    log_weights = log_weights - torch.max(log_weights, dim=int(batch), keepdim=batch)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=int(batch), keepdim=batch)
    weights = torch.exp(log_weights).to(DTYPE)

    mean_new = torch.sum(weights * bar_means, dim=int(batch), keepdim=batch)
    var_new = (torch.sum(weights * bar_means ** 2, dim=int(batch), keepdim=batch) - mean_new ** 2) / scale_factor \
              + torch.sum(weights * bar_vars, dim=int(batch), keepdim=batch)

    var_new = (1. - terminal) * var_new + terminal * 1. / (1. / c_var + scale_factor / (REW_VAR + noise))
    mean_new = (1. - terminal) * mean_new + terminal * var_new * (
            c_mean / c_var + scale_factor * reward / (REW_VAR + noise))

    if torch.isnan(mean_new).any() or torch.isnan(var_new).any():
        import pdb;
        pdb.set_trace()

    if batch:
        return mean_new.squeeze(), torch.squeeze(torch.maximum(torch.tensor(varTH, device=device), var_new)), (
            bar_means, bar_vars, weights)
    else:
        return mean_new, torch.maximum(torch.tensor(varTH, device=device), var_new), (bar_means, bar_vars, weights)


class BayesianDQNAgent(DQNAgent):
    def __init__(self, model, target_model, num_actions, device):
        """Initialize Deep Q-Network agent.

        Parameters
        ----------
        model: torch.nn.Module
            The Q-Network model
        target_model: torch.nn.Module
            The target Q-Network model
        num_actions: int
            Number of possible actions
        device: torch.device
            PyTorch device to use ('cpu' or 'cuda')
        """
        super().__init__(model, target_model, num_actions, device)
        # self.device = device
        # self.model = model.to(device)
        # self.target_model = target_model.to(device)
        # self.num_actions = num_actions

        # Copy weights from model to target_model
        self.update_target_network(1.0)

    def act(self, observation, epsilon=0.0, **kwargs):
        """Select an action based on current observation using epsilon-greedy policy.

        Parameters
        ----------
        observation: numpy.ndarray
            Current observation
        epsilon: float
            Random action probability

        Returns
        -------
        action: int
            Selected action
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.model(obs_tensor)
                q_means = q_values[:, :self.num_actions]
                q_sds = torch.exp(-q_values[:, self.num_actions:])

                # Sample from normal distribution
                samples = torch.randn_like(q_means) * q_sds + q_means
                action = torch.argmax(samples, dim=1)
                return action.numpy()[0]

    # def compute_td_error(self, obs_t, action, reward, obs_tp1, done, gamma):
    #     """Compute TD-error for a single transition.
    #
    #     Parameters
    #     ----------
    #     obs_t: numpy.ndarray
    #         Current observation
    #     action: int
    #         Action taken
    #     reward: float
    #         Reward received
    #     obs_tp1: numpy.ndarray
    #         Next observation
    #     done: bool
    #         Whether the episode is done
    #     gamma: float
    #         Discount factor
    #
    #     Returns
    #     -------
    #     td_error: float
    #         TD-error for the transition
    #     """
    #     obs_t = torch.tensor(obs_t, dtype=torch.float32).unsqueeze(0).to(self.device)
    #     obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32).unsqueeze(0).to(self.device)
    #     action = torch.tensor([action], dtype=torch.int64).to(self.device)
    #     reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
    #     done = torch.tensor([done], dtype=torch.float32).to(self.device)
    #
    #     with torch.no_grad():
    #         q_tp1 = self.target_model(obs_tp1)
    #         q_t_selected = self.model(obs_t).gather(1, action.unsqueeze(1)).squeeze()
    #         q_tp1_best = q_tp1.max(1)[0]
    #         q_tp1_best = (1.0 - done) * q_tp1_best
    #         target = reward + gamma * q_tp1_best
    #         td_error = target - q_t_selected
    #
    #     return td_error.abs().item()

    # def update_target_network(self, tau=1.0):
    #     """Update target network by copying from model.
    #
    #     Parameters
    #     ----------
    #     tau: float
    #         Interpolation parameter - if 1.0, target is set equal to model
    #     """
    #     if tau == 1.0:
    #         self.target_model.load_state_dict(self.model.state_dict())
    #     else:
    #         for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
    #             target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# class ActWrapper(object):
#     def __init__(self, agent, act_params):
#         self._agent = agent
#         self._act_params = act_params
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     @staticmethod
#     def load(path, act_params_new=None):
#         # with open(path, "rb") as f:
#         #     model_data, act_params = cloudpickle.load(f)
#         # if act_params_new:
#         #     for (k, v) in act_params_new.items():
#         #         act_params[k] = v
#         # # Build model
#         # model = models.build_model(**act_params)
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # model.to(device)
#         # with tempfile.TemporaryDirectory() as td:
#         #     arc_path = os.path.join(td, "packed.zip")
#         #     with open(arc_path, "wb") as f:
#         #         f.write(model_data)
#         #     zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
#         #     model.load_state_dict(torch.load(os.path.join(td, "model.pt"), map_location=device))
#         # return ActWrapper(model, act_params)
#         with open(path, "rb") as f:
#             agent, act_params = cloudpickle.load(f)
#             if act_params_new:
#                 for (k, v) in act_params_new.items():
#                     act_params[k] = v
#             return ActWrapper(agent, act_params)
#
#     def __call__(self, observation, stochastic=True, update_eps=-1, **kwargs):
#         epsilon = self._act_params.get('epsilon', 0.1) if update_eps < 0 else update_eps
#         return self._agent.act(observation, epsilon=epsilon * stochastic, **kwargs)
#
#     def save(self, path=None):
#         # if path is None:
#         #     path = os.path.join("model.pkl")
#         # with tempfile.TemporaryDirectory() as td:
#         #     torch.save(self._act.state_dict(), os.path.join(td, "model.pt"))
#         #     arc_name = os.path.join(td, "packed.zip")
#         #     with zipfile.ZipFile(arc_name, 'w') as zipf:
#         #         for root, dirs, files in os.walk(td):
#         #             for fname in files:
#         #                 file_path = os.path.join(root, fname)
#         #                 if file_path != arc_name:
#         #                     zipf.write(file_path, os.path.relpath(file_path, td))
#         #     with open(arc_name, "rb") as f:
#         #         model_data = f.read()
#         # with open(path, "wb") as f:
#         #     cloudpickle.dump((model_data, self._act_params), f)
#         if path is None:
#             path = os.path.join(os.getcwd(), "model.pkl")
#         with open(path, "wb") as f:
#             cloudpickle.dump((self._act, self._act_params), f)
#
#
# def load(path, act_params=None):
#     return ActWrapper.load(path, act_params)


def learn(env,
          q_func,
          lr=5e-4,
          lr_decay_factor=0.99,
          lr_growth_factor=1.01,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=0.9,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          callback=None,
          scope='deepadfq',
          alg='adfq',
          sdMin=1e-5,
          sdMax=1e5,
          noise=0.0,
          act_policy='egreedy',
          epoch_steps=20000,
          eval_logger=None,
          save_dir='.',
          test_eps=0.05,
          init_t=0,
          gpu_memory=1.0,
          render=False,
          reuse_last_init=False,
          **kwargs):
    # device = torch.device("cuda" if torch.cuda.is_available() else
    #                       "mps" if torch.backends.mps.is_available() else
    #                       "cpu")

    # temp for debug, use gpu for results later
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_memory)
    num_actions = env.action_space.n
    observation_space_shape = env.observation_space.shape

    def make_obs_tensor(obs):
        return torch.FloatTensor(np.array(obs)).to(device)

    # Build model
    model = q_func(num_actions * 2).to(device)
    target_model = q_func(num_actions * 2).to(device)
    agent = BayesianDQNAgent(model, target_model, env.action_space.n, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha, device=device)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = max_timesteps
        beta_schedule = linear_schedule(prioritized_replay_beta_iters, prioritized_replay_beta0, 1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size, device=device)
        beta_schedule = None
    exploration = np.ones(max_timesteps)
    exploration[learning_starts + 1:] = np.linspace(1.0, exploration_final_eps, max_timesteps - (learning_starts + 1))
    # ADFQ update function
    adfq_func = posterior_adfq if alg == 'adfq' else posterior_adfq_v2
    act_params = {
        'q_func': q_func,
        'input_shape': observation_space_shape,
        'num_actions': num_actions,
        'act_policy': act_policy,
        'test_eps': test_eps,
        **kwargs
    }
    agent.update_target_network()
    act = ActWrapper(agent, act_params)
    # Training loop
    obs = env.reset()
    t = init_t
    episode_reward = 0
    for t in range(init_t, max_timesteps):
        update_eps = exploration[t]
        action = agent.act(obs,epsilon=update_eps)
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        replay_buffer.add(obs, action, reward, new_obs, float(done))
        obs = new_obs
        episode_reward += reward
        if done:
            obs = env.reset(reuse_last_init=reuse_last_init)
            episode_reward = 0
            if eval_logger:
                eval_logger.log_ep(info)
        if t > learning_starts and (t + 1) % train_freq == 0:
            if prioritized_replay:
                experience = replay_buffer.sample(batch_size, beta=beta_schedule(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            # Get stats from model
            obses_t_tensor = make_obs_tensor(obses_t)
            obses_tp1_tensor = make_obs_tensor(obses_tp1)
            q_stats_t = model(obses_t_tensor)
            q_stats_tp1 = target_model(obses_tp1_tensor)
            ind = torch.arange(batch_size, device=device)
            mean_t = q_stats_t[ind, actions]
            sd_t = torch.exp(-torch.clamp(q_stats_t[ind, actions + num_actions],
                                          -torch.log(torch.tensor(sdMax, device=device)),
                                          -torch.log(torch.tensor(sdMin, device=device))))
            mean_tp1 = q_stats_tp1[:, :num_actions]
            sd_tp1 = torch.exp(
                -torch.clamp(q_stats_tp1[:, num_actions:], -torch.log(torch.tensor(sdMax, device=device)),
                             -torch.log(torch.tensor(sdMin, device=device))))

            target_mean, target_var, _ = adfq_func(
                mean_tp1, torch.square(sd_tp1), mean_t, torch.square(sd_t),
                rewards, gamma, terminal=dones,
                asymptotic=False,
                batch=True, noise=noise, varTH=sdMin * sdMin)

            target_mean = target_mean.reshape(-1)
            target_sd = torch.sqrt(target_var).reshape(-1)
            # Loss: (mean, sd) regression

            criterion = nn.SmoothL1Loss()
            loss = criterion(
                torch.cat((mean_t.view(-1, mean_t.shape[0]).T, sd_t.view(-1, sd_t.shape[0]).T), dim=1),
                torch.cat(
                    (target_mean.view(-1, target_mean.shape[0]).T, target_sd.view(-1, target_sd.shape[0]).T),
                    dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if prioritized_replay:
                new_priorities = torch.abs(mean_t.detach().cpu().numpy() - target_mean.cpu().numpy()) + torch.abs(
                    sd_t.detach().cpu().numpy() - target_sd.cpu().numpy()) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)
            if eval_logger:
                eval_logger.log_step(loss=loss.item())
        if t > learning_starts and (t + 1) % target_network_update_freq == 0:
            target_model.load_state_dict(model.state_dict())
        # if (t + 1) % epoch_steps == 0 and eval_logger:
        #     eval_logger.log_epoch(act)
        if (checkpoint_freq is not None and t > learning_starts and (
                t + 1) % checkpoint_freq == 0 and eval_logger and eval_logger.get_num_episode() > 10):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_{t + 1}.pt"))
        if callback is not None and callback(locals(), globals()):
            break
    # if eval_logger:
    #     eval_logger.finish(max_timesteps, epoch_steps, learning_starts)
    if reuse_last_init:
        with open(os.path.join(save_dir, "init_pose.pkl"), "wb") as f:
            cloudpickle.dump(env.init_pose, f)
    return act
