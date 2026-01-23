import math
import os
from collections import deque

import cloudpickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from ttenv import util
from ttenv.base_model import GMMDist
from ttenv.dqn.utils import plot_gaussian_contours, plot_q_values_heatmap
from ttenv.metadata import METADATA
import faiss
from sklearn.neighbors import KDTree

REG_PARAM = 1e-9
GMM_APPROX_VAR = 1e-3


def project_belief(agent, targets, masking_agent=False):
    if masking_agent:
        res = np.array([])
    else:
        res = np.array(agent)
    for target in targets:
        n = len(target)
        target_dim = len(target[0]) - 1
        gmm_cov = [np.eye(target_dim) * GMM_APPROX_VAR for _ in range(n)]
        gmm = GMMDist(target[:, 0], target[:, 1:], gmm_cov)
        m, v = gmm.compute_mms()
        res = np.concatenate((res, m, np.array(v).flatten()), axis=0)
    return res


def hellinger_distance(m1, v1, m2, v2):
    cov_sum = v1 + v2
    if len(m1) > 1:
        if np.linalg.det(v1) < 0:
            v1 = v1 + REG_PARAM * np.eye(v1.shape[0])
        if np.linalg.det(v2) < 0:
            v2 = v2 + REG_PARAM * np.eye(v2.shape[0])
        squared_distance = 1 - ((np.linalg.det(v1) ** 0.25) * (np.linalg.det(v2) ** 0.25) / np.sqrt(
            np.linalg.det(cov_sum / 2))) * np.exp(
            -1 / 8 * np.dot((m1 - m2), np.dot((m1 - m2), np.linalg.inv(cov_sum / 2))))

        distance = np.sqrt(squared_distance)
    else:
        sigma_1 = np.sqrt(v1)
        sigma_2 = np.sqrt(v2)
        distance = np.sqrt(
            1 - np.sqrt((2 * sigma_1 * sigma_2) / (v1 + v2)) * np.exp(-1 / 4 * ((m1 - m2) ** 2 / (v1 + v2))))
    return distance


class MCKNNModel:
    def __init__(self, target_dim, agent_dim, T, gamma=0.9, target_type="particle", reward_mode="total", n_neighbors=5,
                 n_beliefs=10, post_training=False):
        self.target_dim = target_dim
        self.agent_dim = agent_dim
        # self.gamma = gamma
        self.index = faiss.IndexFlatL2(agent_dim + target_dim)
        self.n_beliefs = n_beliefs
        self.T = T
        self.reward_mode = reward_mode
        self.post_training = post_training

        def distance_metric(data1, data2):
            if target_type == "particle":
                data1_agent = data1[:agent_dim]
                n1 = (len(data1) - agent_dim) // target_dim
                bs1 = np.reshape(data1[agent_dim:], [n1, target_dim])
                m1 = np.average(bs1[:, 1:], axis=0, weights=bs1[:, 0])
                v1 = np.cov(bs1[:, 1:], rowvar=False, aweights=bs1[:, 0])
                data2_agent = data2[:agent_dim]
                n2 = (len(data2) - agent_dim) // target_dim
                bs2 = np.reshape(data2[agent_dim:], [n2, target_dim])
                m2 = np.average(bs2[:, 1:], axis=0, weights=bs2[:, 0])
                v2 = np.cov(bs2[:, 1:], rowvar=False, aweights=bs2[:, 0])
                d = hellinger_distance(m1, v1, m2, v2)
                agent_d = np.linalg.norm(np.array(data1_agent) - np.array(data2_agent))
                return np.sqrt(d ** 2 + agent_d ** 2)
            else:
                raise Exception("Unsupported now")

        self.n_neighbors = n_neighbors
        self.knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance', algorithm="kd_tree")

    def __call__(self, belief_states, agent_info, env_info, knn_state, greedy_flag=False, episode_step=None,
                 save_dir=None, **kwargs):
        print("===== t = " + str(episode_step) + " =====")
        target_beliefs = env_info["target_belief"]
        agent_model = env_info["agent_model"]
        observation_func = env_info["observation_func"]
        action_map = env_info["action_map"]
        q_vals = np.array([0.0 for _ in action_map.values()])
        # if save_dir is not None and not greedy_flag:
        #     all_gaussians = [(elem[self.agent_dim:self.agent_dim + 4],
        #                       np.reshape(elem[self.agent_dim + 4:],
        #                                  [4, 4])) for elem in
        #                      np.array(self.kd_tree_total.data)]
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     plot_gaussian_contours(all_gaussians, saved_path=save_dir + "all points.pdf")
        greedy_r = []
        selected_neighbor_t = {}
        selected_neighbor_e = {}
        future_vs = []
        for i, action in enumerate(action_map.values()):
            next_agent_state,is_col = agent_model.sample_next(action)
            sampled_belief_measurements = [[] for _ in range(self.n_beliefs)]
            obstacles_pt = env_info["MAP"].get_closest_obstacle(next_agent_state)
            if obstacles_pt is None:
                obstacles_pt = (env_info["sensor_r"], np.pi)
            for j, target_belief in enumerate(target_beliefs):
                predictive_zs = target_belief.generate_predictive_samples()
                state_dim = len(predictive_zs[0])
                gmm_cov = [np.eye(state_dim) * GMM_APPROX_VAR for _ in target_belief.weights]
                predictive_gmm = GMMDist(target_belief.weights, predictive_zs, gmm_cov)
                # random_sample_ind = np.random.choice(np.arange(predictive_zs.shape[0]), size=n_belief, replace=True,
                #                                      p=target_belief.weights)
                # selected_zs = predictive_zs[random_sample_ind]
                selected_zs = np.array(predictive_gmm.sample(self.n_beliefs))

                selected_zs = selected_zs[np.random.choice(range(self.n_beliefs), replace=True, size=self.n_beliefs)]
                # selected_z_probs = predictive_gmm.pdf(selected_zs)
                # normalized_selected_z_probs = np.array(selected_z_probs) / np.sum(selected_z_probs)
                sampled_ys = [observation_func(z, next_agent_state) for z in selected_zs]
                sample_next_beliefs = [target_belief.sample_next_belief(y, predictive_zs, next_agent_state) for y in
                                       sampled_ys]

                gmm_next_beliefs = [GMMDist(next_b[0], next_b[1], gmm_cov) for next_b
                                    in
                                    sample_next_beliefs]
                # r_mean = np.mean(
                #     [predictive_gmm.sg_entropy_ub() - next_b.sg_entropy_ub() for next_b
                #      in
                #      gmm_next_beliefs])

                r_mean = np.mean(
                    [predictive_gmm.sg_entropy_ub() - next_b.sg_entropy_ub() for next_b
                     in
                     gmm_next_beliefs]) - is_col
                # r_mean = np.mean(
                #     [-next_b.sg_entropy_ub() for next_b in gmm_next_beliefs])
                # print(r_mean)
                greedy_r.append(r_mean)
                if not greedy_flag and episode_step + 1 != self.T:
                    # assert len(self.values) == len(self.kd_tree.data), "incompatible data size"
                    for k, next_belief in enumerate(sample_next_beliefs):
                        sampled_belief_measurements[k].append([[x] + list(util.relative_distance_polar(y[:2],
                                                                                                       xy_base=next_agent_state[
                                                                                                               :2],
                                                                                                       theta_base=
                                                                                                       next_agent_state[
                                                                                                           2])) + list(
                            util.relative_velocity_polar(
                                y[:2],
                                y[2:],
                                next_agent_state[:2], next_agent_state[2],
                                action[0], action[1])) for x, y in
                                                               zip(next_belief[0], next_belief[1])])
                    if save_dir is not None:
                        gaussians = [(np.array(next_belief.compute_mms()[0]), np.array(next_belief.compute_mms()[1]))
                                     for next_belief in gmm_next_beliefs]
                        save_path = os.path.join(save_dir, str(action) + "/")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        plot_gaussian_contours(gaussians, saved_path=save_path + "sampled beliefs.pdf")
                q_vals[i] = r_mean

            observed_list = []
            observed_list = np.concatenate((observed_list, obstacles_pt))
            observed_list = np.concatenate((observed_list, next_agent_state))
            if not greedy_flag and episode_step + 1 != self.T:
                next_knn_states = [project_belief(observed_list, np.array(next_b)) for next_b in
                                   sampled_belief_measurements]

                # indices, distances = self.kd_tree.query_radius(states, 1.0, return_distance=True)
                if self.reward_mode == "total":
                    # shape of distances (n_belief,k)
                    distances, indices = self.kd_tree_total.query(np.array(next_knn_states), k=self.n_neighbors)
                else:
                    distances, indices = self.kd_trees[episode_step + 1].query(np.array(next_knn_states),
                                                                               k=self.n_neighbors)
                v_futures = []
                idx = 0
                distance_sum = []
                for distance_item, idx_item in zip(distances, indices):
                    distance_sum.append(np.sum(distance_item))
                    normalized_distances = (1 / (np.array(distance_item) + REG_PARAM)) / np.sum(
                        1 / (np.array(distance_item) + REG_PARAM))
                    # distances,indices = self.index.search(states, self.n_neighbors)
                    if self.reward_mode == "total":
                        # v_futures.append(np.average(self.values_total[idx_item, 0], axis=0,
                        #                             weights=normalized_distances))

                        # information rate
                        v_futures.append(
                            np.average(self.values_total[idx_item, 0] / (self.T - self.values_total[idx_item, 1]),
                                       axis=0,
                                       weights=normalized_distances))
                        if action not in selected_neighbor_t.keys():
                            selected_neighbor_t[action] = [self.values_total[idx_item, 1]]
                        else:
                            selected_neighbor_t[action].append(self.values_total[idx_item, 1])

                        if action not in selected_neighbor_e.keys():
                            selected_neighbor_e[action] = [idx_item // self.T]
                        else:
                            selected_neighbor_e[action].append(idx_item // self.T)


                    else:
                        v_futures.append(
                            np.average(self.values[episode_step + 1][idx_item], axis=0,
                                       weights=normalized_distances))
                    if save_dir is not None:
                        gaussians = [(np.array(next_knn_states[idx][self.agent_dim:self.agent_dim + 4]),
                                      np.reshape(next_knn_states[idx][self.agent_dim + 4:],
                                                 [4, 4]))] + [
                                        (item[self.agent_dim:self.agent_dim + 4],
                                         np.reshape(item[self.agent_dim + 4:],
                                                    [4, 4])) for item in
                                        np.array(self.kd_tree_total.data)[idx_item]]
                        save_path = os.path.join(save_dir, str(action) + "/" + "belief_" + str(
                            idx) + "/")
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        plot_gaussian_contours(gaussians, saved_path=save_path + "neighbour_interpolation.pdf")

                    idx += 1
                    # if len(idx_item) >= self.n_neighbors:
                    #     print("neighbour found")
                    #     idx_item = idx_item[:self.n_neighbors]
                    #     distance_item = distance_item[:self.n_neighbors]

                # normalized_distance_sum = (1 / (np.array(distance_sum) + REG_PARAM)) / np.sum(
                #     1 / (np.array(distance_sum) + REG_PARAM))
                # v_future = np.average(v_futures, axis=0,
                #                       weights=normalized_distance_sum)
                v_future = np.mean(v_futures)
                future_vs.append(v_future)
                q_vals[i] += v_future

        print("greedy selection: " + str(np.argmax(greedy_r)))
        if not greedy_flag:
            print("non-myopic selection: " + str(np.argmax(q_vals)))
            if len(selected_neighbor_e.values()) > 0:
                print("future selection: " + str(np.argmax(future_vs)))
                print("selected neighbor at episode:" + str(selected_neighbor_e))
            if len(selected_neighbor_t.values()) > 0:
                print("selected neighbors:" + str(selected_neighbor_t))
        return q_vals

    def fit(self, X, y):
        # build the index
        print("fitting")
        # self.index.add(np.array(X))  # add vectors to the index
        # self.values = np.concatenate((self.values, y), axis=0)
        # self.knn_regressor.fit(X, y)
        if self.reward_mode == "total":
            self.kd_tree_total = KDTree(X, leaf_size=5)
            self.values_total = np.array(y)

        else:
            self.kd_trees = [KDTree(x_item, leaf_size=5) for x_item in X]
            self.values = np.array(y)
        # assert len(self.values) == len(self.kd_tree.data), "incompatible data size"
        print("finish fitting")

    def update_val(self, x, y, radius, episode_step):
        # assert len(self.values) == len(self.kd_tree.data), "incompatible data size"
        if self.reward_mode == "total":
            ind, dists = self.kd_tree_total.query_radius([x], radius, return_distance=True)
        else:
            ind, dists = self.kd_trees[episode_step].query_radius([x], radius, return_distance=True)
        for ind_item, dist_item in zip(ind, dists):
            if len(ind_item) >= self.n_neighbors:
                print("updating state values")
                dist_weights = (1 / (np.array(dist_item) + REG_PARAM)) / np.sum(
                    (1 / (np.array(dist_item) + REG_PARAM)))
                for i, idx in enumerate(ind_item):
                    if self.reward_mode == "total":
                        # self.values_total[idx] = self.values_total[idx] + dist_weights[i] * (
                        #         y - self.values_total[idx])

                        self.values_total[idx][0] = self.values_total[idx][0] + dist_weights[i] * (
                                y[0] - self.values_total[idx][0])
                    else:
                        self.values[episode_step][idx] = self.values[episode_step][idx] + dist_weights[i] * (
                                y - self.values[episode_step][idx])


class MCKNNAgent:
    def __init__(self, model, num_actions, device=torch.device("cpu"), target_model=None):
        """Initialize Deep Q-Network agent.

        Parameters
        ----------
        model: KNN model
        num_actions: int
            Number of possible actions
        device: torch.device
            PyTorch device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.model = model
        self.num_actions = num_actions

    def act(self, observations, epsilon=0.0, **kwargs):
        target_belief = observations["target"]
        agent_info = observations["agent"]
        env_info = observations["env_info"]
        knn_state = observations["knn_state"]
        if np.random.random() < epsilon:
            print("random action")
            return np.random.randint(self.num_actions), None

        q_values = self.model(target_belief, agent_info, env_info, knn_state, **kwargs)
        return np.argmax(q_values), q_values


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

    def save(self, path=None):
        """Save agent to a file.

        Parameters
        ----------
        path: str
            Path to save the agent to
        """
        if path is None:
            path = os.path.join(os.getcwd(), "model.pkl")
        with open(path, "wb") as f:
            cloudpickle.dump((self._agent, self._act_params), f)

    def post_training(self):
        pass


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
          gamma=0.9,
          target_network_update_freq=100,
          param_noise=False,
          callback=None,
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
          blocked=False,
          radius=1.0,
          n_neighbours=5,
          n_beliefs=10,
          reward_mode="total",
          qval_calculation="mc"):
    observation_shape = env.observation_space.shape
    # device = torch.device(
    #     device
    # )
    device = torch.device(
        "cpu"
    )
    debug_mode = False
    qval_calculation = qval_calculation
    num_episodes = 0
    episode_rewards = deque(maxlen=100)
    saved_mean_reward = -math.inf

    exploration_final_eps = 0.0
    if qval_calculation == "mc":
        # exploration = np.ones(max_timesteps)
        # exploration[learning_starts + 1:] = np.linspace(1.0, exploration_final_eps,
        #                                                 max_timesteps - (learning_starts + 1))
        exploration = np.ones(max_timesteps) * exploration_final_eps
    else:
        exploration = np.ones(max_timesteps) * exploration_final_eps

    act_params = {
        'epsilon': exploration_final_eps,
        'num_actions': env.action_space.n,
        'device': device,
        'particle_belief': particle_belief,
        'n_neighbours': n_neighbours,
        'n_beliefs': n_beliefs
    }

    # Initialization variables
    obs = env.reset()
    model = MCKNNModel(len(env.targets[0].state), len(env.agent.state), epoch_steps, gamma=gamma,
                       reward_mode=reward_mode,
                       n_neighbors=n_neighbours,
                       n_beliefs=n_beliefs)
    mcknn_agent = MCKNNAgent(model, env.action_space.n)
    act = ActWrapper(mcknn_agent, act_params)

    env_info = {"action_map": env.action_map, "observation_func": env.sample_observation, "agent_model": env.agent,
                "target_belief": env.belief_targets,"MAP":env.MAP,"sensor_r":env.sensor_r}

    episode_reward = 0
    episode_step = 0
    episode_rewards_history = []
    eval_steps = 1 * epoch_steps
    eval_returns = [[], [], []]
    eval_check = checkpoint_freq // epoch_steps
    eval_episodes = 1
    episode_discovery_rate_dist = []
    lin_dist_range_a2b = METADATA["lin_dist_range_a2b"]
    lin_dist_range_b2t = METADATA["lin_dist_range_b2t"]
    ang_dist_range_a2b = METADATA["ang_dist_range_a2b"]
    if reward_mode == "total":
        belief_states = []
        state_values = []
    else:
        belief_states = [[] for _ in range(epoch_steps)]
        state_values = [[] for _ in range(epoch_steps)]
    step_vals = []
    step_beliefs = []
    step_qvals = []
    if qval_calculation == "mc":
        # Main training loop
        for t in range(max_timesteps):
            # Select action
            # knn_state = np.concatenate((
            #     obs["agent"].cpu().numpy().flatten(), obs["target"].cpu().numpy().flatten()),
            #     axis=0)
            knn_state = project_belief(obs["agent"].cpu().numpy().squeeze(), obs["target"].cpu().numpy())
            step_beliefs.append(knn_state)
            obs["env_info"] = env_info
            obs["knn_state"] = knn_state
            if debug_mode:
                action, q_vals = act(obs, stochastic=True,
                                     update_eps=exploration[min(t, len(exploration) - 1)],
                                     greedy_flag=(t < learning_starts or qval_calculation == "mc-greedy"),
                                     episode_step=episode_step, save_dir=os.path.join(save_dir,
                                                                                      str(num_episodes) + "_debug/t=" + str(
                                                                                          episode_step) + "/"))
            else:
                action, q_vals = act(obs, stochastic=True,
                                     update_eps=exploration[min(t, len(exploration) - 1)],
                                     greedy_flag=(t < learning_starts or qval_calculation == "mc-greedy"),
                                     episode_step=episode_step)
            if q_vals is not None:
                step_qvals.append(q_vals)
            # Execute action and observe next state
            next_obs, reward, terminated, truncated, info = env.step(action)
            # training_dir = os.path.join(save_dir, str(num_episodes) + "_training/")
            # if not os.path.exists(training_dir):
            #     os.makedirs(training_dir)
            # env.render(log_dir=training_dir)
            done = terminated or truncated

            # if t >= learning_starts:
            #     model.update_val(knn_state, v_val, radius, episode_step)
            #     if reward_mode == "total":
            #         belief_states.append(knn_state)
            #         state_values.append(v_val)
            #     else:
            #         belief_states[episode_step].append(knn_state)
            #         state_values[episode_step].append(v_val)
            # else:
            #     step_vals.append(v_val)

            step_vals.append(reward)

            # Update statistics
            episode_reward += reward
            episode_step += 1

            # Update observation
            obs = next_obs

            # End of episode
            if done:
                # Update episode statistics
                print("e=" + str(num_episodes) + " done")
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
                            knn_state = project_belief(obs["agent"].cpu().numpy().squeeze(),
                                                       obs["target"].cpu().numpy())
                            obs["env_info"] = env_info
                            obs["knn_state"] = knn_state
                            action, q_vals = act(obs, stochastic=True,
                                                 update_eps=exploration[min(t, len(exploration) - 1)],
                                                 greedy_flag=(t < learning_starts or qval_calculation == "mc-greedy"),
                                                 episode_step=t_eval)
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

                if qval_calculation == "mc":
                    # if t > learning_starts:
                    #     model.fit(belief_states, state_values)
                    # else:
                    #     for i, bs in enumerate(step_beliefs):
                    #         if reward_mode == "total":
                    #             belief_states.append(bs)
                    #             state_values.append(np.sum(step_vals[i:]))
                    #         else:
                    #             belief_states[i].append(bs)
                    #             state_values[i].append(np.sum(step_vals[i:]))
                    #     if t == learning_starts - 1:
                    #         model.fit(belief_states, state_values)
                    current_val = np.sum(step_vals)
                    for i, bs in enumerate(step_beliefs):
                        if reward_mode == "total":
                            belief_states.append(bs)
                            v = current_val
                            v = (v, i)

                            state_values.append(v)
                            if t >= learning_starts:
                                model.update_val(bs, v, radius, i)

                        else:
                            belief_states[i].append(bs)
                            state_values[i].append(current_val)
                        current_val -= step_vals[i]
                    if t >= learning_starts - 1:
                        model.fit(belief_states, state_values)
                # if len(step_qvals) > 0:
                #     plot_q_values_heatmap(step_qvals, training_dir + "q values.pdf",
                #                           action_labels=np.round(list(env.action_map.values()), 2))
                #     np.savetxt(training_dir + "q_vals.csv", np.array(step_qvals), delimiter=',')
                # Reset environment
                obs = env.reset(reuse_last_init=reuse_last_init, lin_dist_range_a2b=lin_dist_range_a2b,
                                lin_dist_range_b2t=lin_dist_range_b2t, ang_dist_range_a2b=ang_dist_range_a2b,
                                blocked=blocked)
                episode_reward = 0
                episode_step = 0
                step_vals = []
                step_beliefs = []
                step_qvals = []

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

    if reuse_last_init:
        with open(os.path.join(save_dir, "init_pose.pkl"), "wb") as f:
            cloudpickle.dump(env.init_pose, f)
    act.save(checkpoint_path)
    if qval_calculation == "mc":
        print(np.sum([float(sum(dr) > 0) for dr in episode_discovery_rate_dist[-100:]]) / 100.0)
        np.savetxt(save_dir + "_eval_returns.csv", np.array(eval_returns), delimiter=',')
        fig = plt.figure()
        ax = fig.subplots()
        ax.plot(eval_returns[0], eval_returns[1], color='g')
        ax.fill_between(eval_returns[0], np.array(eval_returns[1]) - np.array(eval_returns[2]),
                        np.array(eval_returns[1]) + np.array(eval_returns[2]), color='g',
                        alpha=.3)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        plt.savefig(os.path.join(save_dir, "eval_returns.png"))
        plt.close(fig)

    return act
