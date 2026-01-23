"""
Deep Q-Learning implementation using PyTorch
"""
import math
import os
import tempfile
import zipfile
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, ParticleBeliefReplayBuffer
from ttenv.agent_models import SE2Dynamics
from ttenv.metadata import METADATA
from utils import save_state, load_state
import matplotlib.pyplot as plt

BATCH_SIZE = 128
GAMMA = .9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50000
TAU = .005
LR = 1e-4


class DQNAgent:
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
        self.device = device
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.num_actions = num_actions

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

        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(observation)
        return q_values.argmax(dim=1).item()

    def compute_td_error(self, obs_t, action, reward, obs_tp1, done, gamma):
        """Compute TD-error for a single transition.
        
        Parameters
        ----------
        obs_t: numpy.ndarray
            Current observation
        action: int
            Action taken
        reward: float
            Reward received
        obs_tp1: numpy.ndarray
            Next observation
        done: bool
            Whether the episode is done
        gamma: float
            Discount factor
            
        Returns
        -------
        td_error: float
            TD-error for the transition
        """
        obs_t = torch.tensor(obs_t, dtype=torch.float32).unsqueeze(0).to(self.device)
        obs_tp1 = torch.tensor(obs_tp1, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_tp1 = self.target_model(obs_tp1)
            q_t_selected = self.model(obs_t).gather(1, action.unsqueeze(1)).squeeze()
            q_tp1_best = q_tp1.max(1)[0]
            q_tp1_best = (1.0 - done) * q_tp1_best
            target = reward + gamma * q_tp1_best
            td_error = target - q_t_selected

        return td_error.abs().item()

    def update_target_network(self, tau=1.0):
        """Update target network by copying from model.
        
        Parameters
        ----------
        tau: float
            Interpolation parameter - if 1.0, target is set equal to model
        """
        if tau == 1.0:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class PFDQNAgent:
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
        self.device = device
        self.model = model.to(device)
        self.target_model = target_model.to(device)
        self.num_actions = num_actions

        # Copy weights from model to target_model
        self.update_target_network(1.0)

    def act(self, observations, epsilon=0.0, **kwargs):
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
        target_belief = observations["target"]
        agent_info = observations["agent"]
        # todo: pure greedy, seems not called but tested before, can be deleted
        if "env" in kwargs:
            env = kwargs["env"]
            action_map = env.action_map
            next_target_states = [
                np.average(np.clip(bf.transition_func.sample(bf.states, bf.n), bf.limit[0], bf.limit[1]), axis=0,
                           weights=bf.weights) for
                i, bf in enumerate(env.belief_targets)]
            current_max = -np.inf
            current_max_ac = -1
            agent_info = agent_info.numpy().squeeze()
            for a in action_map.keys():
                next_agent = SE2Dynamics(agent_info[:3], 0.5, action_map[a])
                reward = np.sum(
                    [- np.linalg.norm(np.array(bf[:2]) - np.array(next_agent[:2]))
                     for i, bf in enumerate(next_target_states)])
                if reward > current_max:
                    current_max = reward
                    current_max_ac = a
            return current_max_ac

        with torch.no_grad():
            q_values = self.model(target_belief, agent_info)
        return q_values.argmax(dim=1).item()

    def compute_td_error(self, target_bf_t, agent_info_t, action, reward, target_bf_t1, agent_info_t1, done, gamma):
        """Compute TD-error for a single transition.

        Parameters
        ----------
        target_bf_t: numpy.ndarray
            Current belief state
        agent_info_t:numpy.ndarray
            Current agent info
        action: int
            Action taken
        reward: float
            Reward received
        target_bf_t1: numpy.ndarray
            Next belief state
        agent_info_t1:numpy.ndarray
            Next agent info
        done: bool
            Whether the episode is done
        gamma: float
            Discount factor

        Returns
        -------
        td_error: float
            TD-error for the transition
        """
        action = torch.tensor([action], dtype=torch.int64).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            q_tp1 = self.target_model(target_bf_t1, agent_info_t1)
            q_t_selected = self.model(target_bf_t, agent_info_t).gather(1, action.unsqueeze(1)).squeeze()
            q_tp1_best = q_tp1.max(1)[0]
            q_tp1_best = (1.0 - done) * q_tp1_best
            target = reward + gamma * q_tp1_best
            td_error = target - q_t_selected

        return td_error.abs().item()

    def update_target_network(self, tau=1.0):
        """Update target network by copying from model.

        Parameters
        ----------
        tau: float
            Interpolation parameter - if 1.0, target is set equal to model
        """
        if tau == 1.0:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            target_net_state_dict = self.target_model.state_dict()
            policy_net_state_dict = self.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1.0 - TAU)
            self.target_model.load_state_dict(target_net_state_dict)
            # for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            #     target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


class ActWrapper:
    def __init__(self, agent, act_params):
        """Wrapper for agent's act function.
        
        Parameters
        ----------
        agent: DQNAgent/PFDQNAgent/BayesianDQNAgent
            DQN agent
        act_params: dict
            Parameters for the act function
        """
        self._agent = agent
        self._act_params = act_params

    def __call__(self, observation, stochastic=True, update_eps=-1, **kwargs):
        """Select an action based on the observation.
        
        Parameters
        ----------
        observation: numpy.ndarray
            Current observation
        stochastic: bool
            Whether to use stochastic policy
        update_eps: float
            Epsilon value to use, if negative, uses stored epsilon value
            
        Returns
        -------
        action: int
            Selected action
        """
        epsilon = self._act_params.get('epsilon', 0.1) if update_eps < 0 else update_eps
        return self._agent.act(observation, epsilon=epsilon * stochastic, **kwargs)

    @staticmethod
    def load(path, act_params_new=None, **kwargs):
        """Load agent from file.
        
        Parameters
        ----------
        path: str
            Path to the saved agent
        act_params_new: dict
            New parameters for the act function
            
        Returns
        -------
        act: ActWrapper
            Wrapper for the agent's act function
        """
        with open(path, "rb") as f:
            agent, act_params = cloudpickle.load(f)
            if act_params_new:
                for (k, v) in act_params_new.items():
                    act_params[k] = v
            return ActWrapper(agent, act_params)
        # if act_params_new:
        #     for (k, v) in act_params_new.items():
        #         act_params[k] = v
        #
        # device = act_params.get('device', 'cpu')
        # model = act_params['q_func'](act_params['num_actions']).to(device)
        # target_model = act_params['q_func'](act_params['num_actions']).to(device)
        #
        # with tempfile.TemporaryDirectory() as td:
        #     arc_path = os.path.join(td, "packed.zip")
        #     with open(arc_path, "wb") as f:
        #         f.write(model_data)
        #
        #     zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
        #
        #     state_dict = torch.load(os.path.join(td, "model"), map_location=device)
        #     model.load_state_dict(state_dict['model_state_dict'])
        #     target_model.load_state_dict(state_dict['target_state_dict'])
        #
        #     if act_params["particle_belief"]:
        #         agent = PFDQNAgent(model, target_model, act_params['num_actions'], device)
        #     else:
        #         agent = DQNAgent(model, target_model, act_params['num_actions'], device)

    def save(self, path=None):
        """Save agent to a file.
        
        Parameters
        ----------
        path: str
            Path to save the agent to
        """
        if path is None:
            path = os.path.join(os.getcwd(), "model.pkl")

        # with tempfile.TemporaryDirectory() as td:
        #     save_path = os.path.join(td, "model")
        #     torch.save({
        #         'model_state_dict': self._agent.model.state_dict(),
        #         'target_state_dict': self._agent.target_model.state_dict()
        #     }, save_path)
        #
        #     arc_name = os.path.join(td, "packed.zip")
        #     with zipfile.ZipFile(arc_name, 'w') as zipf:
        #         for root, dirs, files in os.walk(td):
        #             for file in files:
        #                 file_path = os.path.join(root, file)
        #                 if file_path != arc_name:
        #                     zipf.write(file_path, os.path.relpath(file_path, td))
        #
        #     with open(arc_name, "rb") as f:
        #         model_data = f.read()

        with open(path, "wb") as f:
            cloudpickle.dump((self._agent, self._act_params), f)


def load(path, act_params=None):
    """Load agent from file.
    
    Parameters
    ----------
    path: str
        Path to the saved agent
    act_params: dict
        Parameters for the act function
        
    Returns
    -------
    act: ActWrapper
        Wrapper for the agent's act function
    """
    return ActWrapper.load(path, act_params)


def learn(env,
          q_func,
          lr=1e-4,
          lr_decay_factor=0.99,
          lr_growth_factor=1.01,
          max_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=128,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=-1,
          gamma=.9,
          target_network_update_freq=100,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          double_q=False,
          epoch_steps=20000,
          eval_logger=None,
          save_dir='.',
          test_eps=0.05,
          gpu_memory=1.0,
          render=False,
          device="cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else
          "cpu",
          particle_belief=False,
          reuse_last_init=False,
          blocked=False):
    """Train a Deep Q-Network model.
    
    Parameters
    ----------
    env: gym.Env
        Environment to train on
    q_func: function
        Function that takes observation_shape and num_actions and returns a torch.nn.Module
    lr: float
        Learning rate for optimizer
    lr_decay_factor: float
        Factor to decrease learning rate by
    lr_growth_factor: float
        Factor to increase learning rate by
    max_timesteps: int
        Number of env steps to optimize for
    buffer_size: int
        Size of the replay buffer
    exploration_fraction: float
        Fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        Final value of random action probability
    train_freq: int
        Update the model every `train_freq` steps
    batch_size: int
        Size of batches sampled from replay buffer for training
    print_freq: int
        How often to print out training progress
    checkpoint_freq: int
        How often to save the model
    checkpoint_path: str
        Path to save the model
    learning_starts: int
        How many steps of the model to collect transitions for before learning starts
    gamma: float
        Discount factor
    target_network_update_freq: float
        Update the target network every `target_network_update_freq` steps
    prioritized_replay: bool
        If True, use prioritized replay buffer
    prioritized_replay_alpha: float
        Alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        Initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        Number of iterations over which beta will be annealed
    prioritized_replay_eps: float
        Epsilon to add to the TD errors when updating priorities
    param_noise: bool
        Whether to use parameter noise
    callback: function
        Called every step with state of the algorithm
    double_q: bool
        Whether to use double Q-learning
    epoch_steps: int
        Number of steps per epoch
    eval_logger: Logger
        Logger for evaluation
    save_dir: str
        Path for saving results
    test_eps: float
        Epsilon value for testing
    gpu_memory: float
        Fraction of GPU memory to use
    render: bool
        Whether to render the environment
    device: str
        PyTorch device to use
    particle_belief: bool
        If using particle belief
    reuse_last_init: bool
        If reuse the initialization pose all the time
    blocked: bool
        If the initialized target is blocked
    Returns
    -------
    act: ActWrapper
        Wrapper around act function
    """
    # Create optimizers
    observation_shape = env.observation_space.shape
    device = torch.device(
        device
    )
    # Create target model
    model = q_func(env.action_space.n).to(device)
    target_model = q_func(env.action_space.n).to(device)
    if particle_belief:
        agent = PFDQNAgent(model, target_model, env.action_space.n, device)
        replay_buffer = ParticleBeliefReplayBuffer(buffer_size, device=device)
    else:
        agent = DQNAgent(model, target_model, env.action_space.n, device)

        # Create replay buffer
        if prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, device=device)
            if prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = max_timesteps
            beta_schedule = np.linspace(prioritized_replay_beta0, 1.0, prioritized_replay_beta_iters)
        else:
            replay_buffer = ReplayBuffer(buffer_size, device=device)
            beta_schedule = None

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # Initialize variables
    num_episodes = 0
    episode_rewards = deque(maxlen=100)
    saved_mean_reward = -math.inf

    # Function to train the model
    def train():
        nonlocal saved_mean_reward

        if len(replay_buffer) < batch_size:
            return 0
        if particle_belief:
            batch = replay_buffer.sample(batch_size)
            target_bfs_t, agent_info_t, actions, rewards, target_bfs_t1, agent_info_t1, dones = batch
            weights, batch_idxes = torch.tensor(np.ones_like(rewards.cpu()), device=device, dtype=torch.float32), None
            q_t = model(target_bfs_t, agent_info_t)
        else:
            # Sample from replay buffer
            if prioritized_replay:
                beta = beta_schedule[min(t, prioritized_replay_beta_iters - 1)]
                batch = replay_buffer.sample(batch_size, beta)
                obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes = batch
            else:
                batch = replay_buffer.sample(batch_size)
                obses_t, actions, rewards, obses_tp1, dones = batch
                weights, batch_idxes = torch.tensor(np.ones_like(rewards.cpu()), device=device,
                                                    dtype=torch.float32), None
            q_t = model(obses_t)
        # Compute current Q-values
        q_t_selected = q_t.gather(1, actions.unsqueeze(1))

        # Compute next Q-values using target network
        with torch.no_grad():
            if double_q:
                # Double Q-learning: Select actions using the main network
                if particle_belief:
                    q_tp1 = model(target_bfs_t1, agent_info_t1)
                else:
                    q_tp1 = model(obses_tp1)
                _, a_tp1 = q_tp1.max(dim=1)

                # Evaluate those actions using the target network
                q_tp1_target = target_model(obses_tp1)
                q_tp1_best = q_tp1_target.gather(1, a_tp1.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: Select best actions using the target network
                if particle_belief:
                    q_tp1 = model(target_bfs_t1, agent_info_t1)
                else:
                    q_tp1 = model(obses_tp1)
                q_tp1_best = q_tp1.max(1)[0]
            # Zero out Q-values for terminal states
            q_tp1_best = q_tp1_best * (1.0 - dones)

            # Compute the TD target
            td_target = rewards + gamma * q_tp1_best

        # Compute Huber loss
        td_error = q_t_selected - td_target
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_t_selected, td_target.unsqueeze(1))

        # loss = F.smooth_l1_loss(q_t_selected, td_target, reduction='none')

        # Apply importance weights from prioritized replay
        # weighted_loss = torch.mean(loss * weights)

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        # for param in model.parameters():
        #     param.grad.data.clamp_(-1, 1)
        torch.nn.utils.clip_grad_value_(model.parameters(), 100)
        optimizer.step()
        # scheduler.step()

        # Update target network if it's time
        if target_network_update_freq < 1:
            # Soft target update
            agent.update_target_network(target_network_update_freq)
        else:
            # Hard target update
            if t != 0 and t % target_network_update_freq == 0:
                agent.update_target_network(TAU)

        # Update priorities in prioritized replay buffer
        if prioritized_replay:
            new_priorities = np.abs(td_error.detach().cpu().numpy()) + prioritized_replay_eps
            replay_buffer.update_priorities(batch_idxes, new_priorities)

        return (loss * weights).mean().item()

    # Create the schedule for exploration
    exploration = np.ones(max_timesteps)
    exploration[learning_starts + 1:] = np.linspace(1.0, exploration_final_eps, max_timesteps - (learning_starts + 1))
    # steps_left = max_timesteps - (learning_starts + 1)
    # exploration[learning_starts + 1:] = np.array(
    #     [exploration_final_eps + (1.0 - exploration_final_eps) * math.exp(-1. * steps_done / EPS_DECAY) for steps_done
    #      in
    #      range(steps_left)])

    # Initialize the parameters and copy them to the target network
    agent.update_target_network()

    # Create action function
    act_params = {
        'epsilon': exploration_final_eps,
        'q_func': q_func,
        'num_actions': env.action_space.n,
        'device': device,
        'particle_belief': particle_belief
    }
    act = ActWrapper(agent, act_params)

    # Initialization variables
    obs = env.reset()
    episode_reward = 0
    episode_step = 0
    loss = 0
    t = 0
    episode_rewards_history = []
    eval_steps = 1 * epoch_steps
    eval_returns = [[], [], []]
    eval_check = checkpoint_freq // epoch_steps
    eval_episodes = 1
    episode_discovery_rate_dist = []
    lin_dist_range_a2b = METADATA["lin_dist_range_a2b"]
    lin_dist_range_b2t = METADATA["lin_dist_range_b2t"]
    ang_dist_range_a2b = METADATA["ang_dist_range_a2b"]
    add_times = 0
    # Main training loop
    for t in range(max_timesteps):
        # Select action

        action = act(obs, stochastic=True, update_eps=exploration[min(t, len(exploration) - 1)])

        # Execute action and observe next state
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        # Store transition in the replay buffer
        if particle_belief:
            replay_buffer.add(obs["target"], obs["agent"], action, reward, next_obs["target"], next_obs["agent"],
                              float(done))
        else:
            replay_buffer.add(obs, action, reward, next_obs, float(done))

        # Update statistics
        episode_reward += reward
        episode_step += 1

        # Train the network
        if t > learning_starts and t % train_freq == 0:
            loss = train()

        # Update observation
        obs = next_obs

        # End of episode
        if done:
            # Update episode statistics

            episode_rewards.append(episode_reward)
            episode_rewards_history.append(episode_reward)
            episode_discovery_rate_dist.append([dr / epoch_steps for dr in env.discover_cnt])
            # if num_episodes % (checkpoint_freq // epoch_steps) == 0:
            if num_episodes % eval_check == 0:
                rollout_dir = os.path.join(save_dir, str(num_episodes) + "_eval_rollout/")
                if not os.path.exists(rollout_dir):
                    os.makedirs(rollout_dir)

                eval_returns[0].append(num_episodes)
                eval_episode_rewards = []
                for e_e in range(eval_episodes):
                    eval_episode_reward = 0
                    obs = env.reset(reuse_last_init=reuse_last_init, lin_dist_range_a2b=lin_dist_range_a2b,
                                    lin_dist_range_b2t=lin_dist_range_b2t, ang_dist_range_a2b=ang_dist_range_a2b,
                                    blocked=blocked)
                    for t_eval in range(int(eval_steps)):
                        action = act(obs, stochastic=True, update_eps=0.0)

                        # Execute action and observe next state
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        eval_episode_reward += reward
                        obs = next_obs
                        if e_e == eval_episodes - 1:
                            env.render(log_dir=rollout_dir)
                    eval_episode_rewards.append(eval_episode_reward)
                eval_returns[1].append(np.mean(eval_episode_rewards))
                eval_returns[2].append(np.std(eval_episode_rewards))
            # temp code, for training
            # if num_episodes % 100 == 0 and np.mean(episode_discovery_rate_dist[-100:]) > .8:
            #     if reuse_last_init:
            #         env.init_pose["targets"][0][0] = np.clip(env.init_pose["targets"][0][0] + 1, env.MAP.mapmin[0],
            #                                                  env.MAP.mapmax[0] - 1.0)
            #         env.init_pose["targets"][0][1] = np.clip(env.init_pose["targets"][0][1] + 1, env.MAP.mapmin[1],
            #                                                  env.MAP.mapmax[1] - 1.0)
            #     add_times += 1
            #     lin_dist_range_a2b = (lin_dist_range_a2b[0], min(20.0, lin_dist_range_a2b[1] + add_times * 1.0))
            #     lin_dist_range_b2t = (lin_dist_range_b2t[0], min(20.0, lin_dist_range_b2t[1] + add_times * 1.0))
            #     ang_dist_range_a2b = (
            #         max(-np.pi, ang_dist_range_a2b[0] - add_times * .1),
            #         min(np.pi, ang_dist_range_a2b[1] + add_times * .1))
            #     speed = min(env.target_speed_limit + 1.0, 3.0)
            #     env.set_limits(target_speed_limit=speed)
            #     env.init_pose["targets"][0][2] = speed
            #     env.targets[0].limit = env.limit['target']
            num_episodes += 1
            # Reset environment
            obs = env.reset(reuse_last_init=reuse_last_init, lin_dist_range_a2b=lin_dist_range_a2b,
                            lin_dist_range_b2t=lin_dist_range_b2t, ang_dist_range_a2b=ang_dist_range_a2b,
                            blocked=blocked)
            episode_reward = 0
            episode_step = 0

        # Evaluate and save model
        if t > learning_starts and checkpoint_freq is not None and t % checkpoint_freq == 0:
            # Compute mean reward
            mean_100ep_reward = np.mean(episode_rewards)

            # Print progress
            if print_freq is not None and len(episode_rewards) > 1 and t % print_freq == 0:
                print(f"Steps: {t}")
                print(f"Episodes: {num_episodes}")
                print(f"Mean 100 episode reward: {mean_100ep_reward:.2f}")
                print(f"% time spent exploring: {int(100 * exploration[min(t, len(exploration) - 1)])}")
                print(f"Learning rate: {lr:.5f}")

            # Save best model
            if checkpoint_path is not None:
                if mean_100ep_reward > saved_mean_reward:
                    print(
                        f"mean reward increase: {saved_mean_reward:.2f} -> {mean_100ep_reward:.2f}")
                    act.save(checkpoint_path)
                    saved_mean_reward = mean_100ep_reward

        # Call callback if provided
        if callback is not None:
            if callback(locals(), globals()):
                break
    if reuse_last_init:
        with open(os.path.join(save_dir, "init_pose.pkl"), "wb") as f:
            cloudpickle.dump(env.init_pose, f)
    # plt.plot(np.arange(0,num_episodes),episode_last_step_dist)
    # plt.savefig(os.path.join(save_dir, "last step distance.png"))
    if np.mean(episode_rewards) > saved_mean_reward:
        act.save(checkpoint_path)
    print(np.sum([float(sum(dr) > 0) for dr in episode_discovery_rate_dist[-100:]]) / 100.0)
    np.savetxt(save_dir + "_eval_returns.csv", np.array(eval_returns), delimiter=',')
    fig = plt.figure()
    ax = fig.subplots()
    ax.errorbar(eval_returns[0], eval_returns[1], yerr=eval_returns[2], fmt='-x', color='g', capsize=5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    plt.savefig(os.path.join(save_dir, "eval_returns.png"))
    plt.close(fig)
    return act
