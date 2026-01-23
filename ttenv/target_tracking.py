"""Target Tracking Environments for Reinforcement Learning. OpenAI gym format

[Vairables]

d: radial coordinate of a belief target in the learner frame
alpha : angular coordinate of a belief target in the learner frame
ddot : radial velocity of a belief target in the learner frame
alphadot : angular velocity of a belief target in the learner frame
Sigma : Covariance of a belief target
o_d : linear distance to the closet obstacle point
o_alpha : angular distance to the closet obstacle point

[Environment Descriptions]

TargetTrackingEnv0 : Static Target model + noise - No Velocity Estimate
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets , [o_d, o_alpha]
    Target: Static [x,y] + noise
    Belief Target: KF, Estimate only x and y

TargetTrackingEnv0_1 : Static Target model + noise - No Velocity Estimate
    RL state: [particle belief, observed] * nb_targets, [x, o_d, o_alpha]
    Target: Static [x,y] + noise
    Belief Target : PF, Estimate only x and y

TargetTrackingEnv1 : Double Integrator Target model with KF belief tracker
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : Nonlinear Double Integrator model, [x,y,xdot,ydot]
    Belief Target : KF, Double Integrator model

TargetTrackingEnv1_1 : Double Integrator Target model with PF belief tracker [x,y,theta,v,w]
    RL state: [particle belief] * nb_targets, x
    Target : Nonlinear Double Integrator model, [x,y,xdot,ydot]
    Belief Target : PF, Double Integrator model

TargetTrackingEnv2 : SE2 Target model with UKF belief tracker
    RL state: [d, alpha, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2 model [x,y,theta]

TargetTrackingEnv3 : SE2 Target model with UKF belief tracker [x,y,theta,v,w]
    RL state: [d, alpha, ddot, alphadot, logdet(Sigma), observed] * nb_targets, [o_d, o_alpha]
    Target : SE2 model [x,y,theta] + a control policy u=[v,w]
    Belief Target : UKF for SE2Vel model [x,y,theta,v,w]



"""
import torch
from gym import spaces, logger

import numpy as np
from numpy import linalg as LA

from ttenv.base import TargetTrackingBase
from ttenv.base_model import GMMDist, LinearGaussianDistribution, GaussianDistribution, GasLeakageModel
from ttenv.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief, UKFbelief, PFbelief, AuxiliaryPFbelief
from ttenv.metadata import METADATA, DEVICE, LEAKAGE_OBS
import ttenv.util as util
from ttenv.base import TargetTrackingBase


class TargetTrackingEnv0(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                 known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
                                    is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v0'
        self.target_dim = 2

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def reset(self, **kwargs):
        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                init_state=init_pose['belief_targets'][i][:self.target_dim],
                init_cov=self.target_init_cov)
            self.targets[i].reset(np.array(init_pose['targets'][i][:self.target_dim]))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                self.belief_targets[i].state[:2],
                xy_base=self.agent.state[:2],
                theta_base=self.agent.state[2])
            self.state.extend([r_b, alpha_b,
                               np.log(LA.det(self.belief_targets[i].cov)),
                               float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit map for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self):
        self.num_target_dep_vars = 4
        self.num_target_indep_vars = 2

        self.limit = {}  # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                              collision_func=lambda x: self.MAP.is_collision(x))

        self.target_noise_cov = self.const_q * self.sampling_period ** 3 / 3 * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.eye(2)
        self.targetA = np.eye(self.target_dim)

        # Build a target
        self.targets = [AgentDoubleInt2D(dim=self.target_dim, sampling_period=self.sampling_period,
                                         limit=self.limit['target'],
                                         collision_func=lambda x: self.MAP.is_collision(x),
                                         A=self.targetA, W=self.target_true_noise_sd) for _ in range(self.num_targets)]
        self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
                                        W=self.target_noise_cov, obs_noise_func=self.observation_noise,
                                        collision_func=lambda x: self.MAP.is_collision(x))
                               for _ in range(self.num_targets)]


class TargetTrackingEnv0_1(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True,
                 known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
                                    is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v0.1'
        self.target_dim = 2

        # Set limits.
        self.set_limits()
        self.n_particles = 1000 if 'n_particles' not in kwargs else kwargs['n_particles']
        self.alpha = 0.5
        self.epsilon = .05
        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def _generate_prior_dist(self, init_pose):
        # var = .5
        # # init_pose = super().reset()
        # dists = []
        # for i in range(self.num_targets):
        #     # init_mean = init_pose['belief_targets'][i][:self.target_dim]
        #     x_size = [self.MAP.mapmin[0], self.MAP.mapmax[0]]
        #     y_size = [self.MAP.mapmin[1], self.MAP.mapmax[1]]
        #     # x_size = [init_mean[0]-np.sqrt(self.target_init_cov), self.MAP.mapmax[0]+np.sqrt(self.target_init_cov)]
        #     # y_size = [self.MAP.mapmin[1]-np.sqrt(self.target_init_cov), self.MAP.mapmax[1]+np.sqrt(self.target_init_cov)]
        #     n_bins = 1000
        #     x = np.linspace(x_size[0], x_size[1], n_bins)
        #     y = np.linspace(y_size[0], y_size[1], n_bins)
        #     xx, yy = np.meshgrid(x, y)
        #     positions = np.vstack([xx.ravel(), yy.ravel()]).T
        #     prior_gmm = GMMDist([1 / (n_bins ** 2)] * n_bins ** 2, positions,
        #                         np.array([np.diag([var] * target_dim)] * n_bins ** 2))
        #     dists.append(prior_gmm)
        # return dists
        dists = [GaussianDistribution(np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                                      np.eye(self.target_dim) * self.target_init_cov) for i in range(self.num_targets)]
        return dists
        # init_pose = super().reset()
        # dists = [GaussianDistribution(init_pose['belief_targets'][i][:self.target_dim],np.eye(self.target_dim)*self.target_init_cov) for i in range(self.num_targets)]
        # return dists

    def reset(self, **kwargs):
        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)

        self.agent.reset(init_pose['agent'])
        if 'reuse_last_init' not in kwargs or not kwargs['kwargs']:
            self.prior_dists = self._generate_prior_dist(init_pose)

        for i in range(self.num_targets):
            self.belief_targets[i].reset(self.prior_dists[i])
            self.targets[i].reset(np.array(init_pose['targets'][i][:self.target_dim]))
        self.has_discovered = [0] * self.num_targets
        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()
        self.last_est_dists = [np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for bf in
                               self.belief_targets]
        self.last_ents = [bf.entropy() for bf in self.belief_targets]
        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = {}
        particle_list = []
        observed_list = []
        for i in range(self.num_targets):
            # r_b, alpha_b = util.relative_distance_polar(
            #     self.belief_targets[i].state[:2],
            #     xy_base=self.agent.state[:2],
            #     theta_base=self.agent.state[2])
            particle_list += [np.append(y, x) for x, y in
                              zip(self.belief_targets[i].weights, self.belief_targets[i].states)]
            observed_list.append(float(observed[i]))
        observed_list = np.concatenate((observed_list, self.agent.state))
        observed_list = np.concatenate((observed_list, obstacles_pt))
        self.state = {"target": torch.tensor(np.array(particle_list), dtype=torch.float32, device=DEVICE).unsqueeze(0),
                      "agent": torch.tensor(np.array([observed_list]), dtype=torch.float32, device=DEVICE).unsqueeze(0)}

        # Update the visit map for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self):
        self.num_target_dep_vars = 4
        self.num_target_indep_vars = 2

        self.limit = {}  # 0: low, 1:high
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [self.MAP.mapmin, self.MAP.mapmax]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def observation(self, target):
        r, alpha = util.relative_distance_polar(target.state[:2],
                                                xy_base=self.agent.state[:2],
                                                theta_base=self.agent.state[2])
        observed = (r <= self.sensor_r) \
                   & (abs(alpha) <= self.fov / 2 / 180 * np.pi) \
                   & (not (self.MAP.is_blocked(self.agent.state, target.state)))
        z = np.array([r, alpha])
        z += np.random.multivariate_normal(np.zeros(2, ), self.observation_noise(observed, target=target.state[:2],
                                                                                 agent=self.agent.state[:2]))

        return observed, z

    def observe_and_update_belief(self):
        observed = []
        for i in range(self.num_targets):
            observation = self.observation(self.targets[i])
            observed.append(observation[0])

            if observation[0]:  # if observed, update the target belief.
                self.belief_targets[i].update(observation, self.agent.state)
                # self.belief_targets[i].update(observation, self.agent.state)
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
        return observed

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                              collision_func=lambda x: self.MAP.is_collision(x))

        self.target_noise_cov = self.const_q * self.sampling_period ** 3 / 3 * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = METADATA['const_q_true'] * np.eye(2)
        self.targetA = np.eye(self.target_dim)
        self.target_transition = LinearGaussianDistribution(coefficient=self.targetA,
                                                            noise_mean=np.zeros(self.target_dim),
                                                            noise_var=self.target_true_noise_sd)
        # Build a target
        self.targets = [AgentDoubleInt2D(dim=self.target_dim, sampling_period=self.sampling_period,
                                         limit=self.limit['target'],
                                         collision_func=lambda x: self.MAP.is_collision(x),
                                         A=self.targetA, W=self.target_true_noise_sd) for _ in range(self.num_targets)]
        # self.belief_targets = [KFbelief(dim=self.target_dim, limit=self.limit['target'], A=self.targetA,
        #                     W=self.target_noise_cov, obs_noise_func=self.observation_noise,
        #                     collision_func=lambda x: self.MAP.is_collision(x))
        #                         for _ in range(self.num_targets)]

        # self.belief_targets = [
        #     PFbelief(dim=self.target_dim, limit=self.limit['target'], transition_func=self.target_transition,
        #              n=self.n_particles, effective_n=self.n_particles // 2, dim_z=2,
        #              obs_noise_func=self.observation_noise,
        #              collision_func=lambda x: self.MAP.is_collision(x))
        #     for i in range(self.num_targets)]

        self.belief_targets = [
            PFbelief(dim=self.target_dim, limit=self.limit['target'], transition_func=self.target_transition,
                     n=self.n_particles, effective_n=self.n_particles // 2, dim_z=2,
                     obs_noise_func=self.observation_noise,
                     collision_func=lambda x: self.MAP.is_collision(x))
            for i in range(self.num_targets)]

    # modify for other reward, e.g., GMM MI estimation/NMC
    # def get_reward(self, is_training=True, **kwargs):
    #     # reward = self.last_est_dist - np.sum([ np.linalg.norm(np.array(bf.state[:self.target_dim]) - np.array(self.agent.state[:self.target_dim])) for bf in self.belief_targets])
    #     reward = np.sum([(self.last_ents[i] - bf.entropy()) / np.linalg.norm(
    #         np.array(bf.state[:self.target_dim]) - np.array(self.agent.state[:self.target_dim])) for i, bf in
    #                      enumerate(self.belief_targets)])
    #     return reward, False, 0, 0

    def get_reward(self, is_training=True, **kwargs):
        c_mean = 0.1
        c_std = 0.0
        c_penalty = 1.0
        detcov = np.array([LA.det(b_target.cov) for b_target in self.belief_targets])
        detcov[np.where(detcov <= 0)] = 10 ** -9
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        reward = c_mean * r_detcov_mean + c_std * r_detcov_std
        reward = min(0.0, reward) - c_penalty * 1.0

        return reward, False, r_detcov_mean, r_detcov_std


class TargetTrackingEnv1(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
                                    is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1'
        self.target_dim = 4
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits(target_speed_limit=METADATA['target_speed_limit'])

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        self.discover_cnt = [0 for _ in range(self.num_targets)]
        self.agent_target_dist = [[] for _ in range(self.num_targets)]
        self.max_ent = np.log(np.prod(np.array(self.limit["target"][1]) - np.array(self.limit["target"][0])))

        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                init_cov=self.target_init_cov)
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def step(self, action):
        # The agent performs an action (t -> t+1)
        action_vw = self.action_map[action]
        is_col = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        self.num_collisions += int(is_col)

        # The targets move (t -> t+1)
        for i in range(self.num_targets):
            if self.has_discovered[i]:
                self.targets[i].update(self.agent.state[:2])

        self.last_est_dists = [np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for bf in
                               self.belief_targets]
        for i in range(self.num_targets):
            self.agent_target_dist[i].append(
                np.linalg.norm(np.array(self.targets[i].state[:2]) - np.array(self.agent.state[:2])))
        self.last_ents = [bf.entropy() for bf in self.belief_targets]
        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training,
                                                                        is_col=is_col, observed=observed)

        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func(action_vw, observed)

        return self.state, reward, False, False, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def observe_and_update_belief(self):
        observed = []
        for i in range(self.num_targets):
            observation = self.observation(self.targets[i])
            observed.append(observation[0])
            if observation[0]:  # if observed, update the target belief.
                self.belief_targets[i].update(observation[1], self.agent.state)
                self.discover_cnt[i] += 1
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
        return observed

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                        xy_base=self.agent.state[:2],
                                                        theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                self.belief_targets[i].state[:2],
                self.belief_targets[i].state[2:],
                self.agent.state[:2], self.agent.state[2],
                action_vw[0], action_vw[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                               np.log(LA.det(self.belief_targets[i].cov)),
                               float(observed[i])])
            # self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  # Maximum relative speed

        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(
            ([0.0, -np.pi, -rel_speed_limit, -10 * np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
            np.concatenate((
                [600.0, np.pi, rel_speed_limit, 10 * np.pi, 50.0, 2.0] * self.num_targets,
                [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        # self.observation_space = spaces.Box(self.limit['state'][0][:4], self.limit['state'][1][:4], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                              collision_func=lambda x: self.MAP.is_collision(x),
                              policy=CirclePolicy(self.sampling_period, self.MAP.origin, 3.0))
        # Build targets
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                                       self.sampling_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.sampling_period ** 3 / 3 * np.eye(2),
                            self.sampling_period ** 2 / 2 * np.eye(2)), axis=1),
            np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                            self.sampling_period * np.eye(2)), axis=1)))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * np.concatenate((
                np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                                self.sampling_period / 2 * np.eye(2)), axis=1),
                np.concatenate((self.sampling_period / 2 * np.eye(2),
                                self.sampling_period * np.eye(2)), axis=1)))

        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                                                   self.sampling_period, self.limit['target'],
                                                   lambda x: self.MAP.is_collision(x),
                                                   W=self.target_true_noise_sd, A=self.targetA,
                                                   obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                                       x, fov=2 * np.pi, r_max=10e2))
                        for _ in range(self.num_targets)]
        if self.observation_model == LEAKAGE_OBS:
            self.belief_targets = [UKFbelief(dim=self.target_dim,
                                             limit=self.limit['target'], fx=self.targets[i].fx,
                                             W=self.target_noise_cov,
                                             hx=lambda target_coord, agent_coord: np.array(
                                                 [self.leakage_model.generate_sample(target_coord, agent_coord)]),
                                             obs_noise_func=self.observation_noise,
                                             collision_func=lambda x: self.MAP.is_collision(x),
                                             measurement_model=self.observation_model)
                                   for i in range(self.num_targets)]

            # self.belief_targets = [KFbelief(dim=self.target_dim,
            #                                 limit=self.limit['target'], A=self.targetA,
            #                                 W=self.target_noise_cov,
            #                                 obs_noise_func=self.observation,
            #                                 collision_func=lambda x: self.MAP.is_collision(x))
            #                        for _ in range(self.num_targets)]
        else:
            self.belief_targets = [KFbelief(dim=self.target_dim,
                                            limit=self.limit['target'], A=self.targetA,
                                            W=self.target_noise_cov,
                                            obs_noise_func=self.observation_noise,
                                            collision_func=lambda x: self.MAP.is_collision(x))
                                   for _ in range(self.num_targets)]

    def get_reward(self, is_training=True, **kwargs):
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))
        c_mean = 0.1
        c_std = 0.0
        c_penalty = 1.0
        reward = c_mean * r_detcov_mean + c_std * r_detcov_std
        if "is_col" in kwargs.keys() and kwargs["is_col"]:
            reward = min(0.0, reward) - c_penalty * 1.0
        return reward, False, r_detcov_mean, r_detcov_std
        # shaped reward
        # c_penalty = 1
        # detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        # normed_ent_reward = - np.mean(np.log(detcov))/self.max_ent
        # ob_reward = np.sum([float(ob) for ob in kwargs["observed"]])
        # reward = normed_ent_reward + ob_reward
        # # reward = np.sum([float(ob) for ob in kwargs["observed"]])
        # if "is_col" in kwargs.keys() and kwargs["is_col"]:
        #     reward = reward - 1.0*c_penalty
        # return reward,False, 0, 0

        # reward = np.sum([(self.last_ents[i] - bf.entropy()) / np.linalg.norm(
        #     np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for i, bf in
        #                  enumerate(self.belief_targets)])
        # return reward, False, 0, 0
    #     # reward = np.sum([float(ob) for ob in kwargs["observed"]])
    #     # max_d = np.sqrt(
    #     #     (self.MAP.mapmax[0] - self.MAP.mapmin[0]) ** 2 + (self.MAP.mapmax[1] - self.MAP.mapmin[1]) ** 2)
    #     # reward = np.sum([1 - LA.norm(np.array(bs.state[:2]) - np.array(self.agent.state[:2]))/max_d for bs in self.targets])
    #     # within_range = [LA.norm(np.array(bs.state[:2]) - np.array(self.agent.state[:2])) > self.sensor_r for i, bs in
    #     #                 self.targets]
    #     xy_target_base = [util.transform_2d(bs.state[:2], self.agent.state[2], self.agent.state[:2]) for bs in
    #                       self.belief_targets]
    #     min_dist_to_perfect = .05
    #     m_reward = 1.0 / min_dist_to_perfect
    #     reward = np.sum(
    #         [m_reward if ob else 1 / np.clip(LA.norm(np.array(xy) - np.array([5, 0])), min_dist_to_perfect, np.inf) for
    #          ob, xy in
    #          zip(kwargs["observed"], xy_target_base)])
    #     return reward, False, 0, 0


class TargetTrackingEnv1_1(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
                                    is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v1'
        self.target_dim = 4
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits(target_speed_limit=METADATA['target_speed_limit'])
        self.n_particles = 1000 if 'n_particles' not in kwargs else kwargs['n_particles']
        self.alpha = 0.5
        self.epsilon = .05
        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)
        self.discover_cnt = [0 for _ in range(self.num_targets)]

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        self.discover_cnt = [0 for _ in range(self.num_targets)]
        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        if 'reuse_last_init' not in kwargs or not kwargs['reuse_last_init']:
            self.prior_dists = self._generate_prior_dist(init_pose)
        self.agent.reset(init_pose['agent'])

        for i in range(self.num_targets):
            self.belief_targets[i].reset(self.prior_dists[i])
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()
        self.last_est_dists = [np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for bf in
                               self.belief_targets]
        self.last_ents = [bf.entropy() for bf in self.belief_targets]
        self.agent_target_dist = [[] for _ in range(self.num_targets)]
        self.max_ent = np.log(np.prod(np.array(self.limit["target"][1]) - np.array(self.limit["target"][0])))
        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        return self.state

    def step(self, action):
        # The agent performs an action (t -> t+1)
        action_vw = self.action_map[action]
        is_col = self.agent.update(action_vw, [t.state[:2] for t in self.targets])
        self.num_collisions += int(is_col)

        # The targets move (t -> t+1)
        for i in range(self.num_targets):
            if self.has_discovered[i]:
                self.targets[i].update(self.agent.state[:2])

        self.last_est_dists = [np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for bf in
                               self.belief_targets]
        self.last_ents = [bf.entropy() for bf in self.belief_targets]
        for i in range(self.num_targets):
            self.agent_target_dist[i].append(
                np.linalg.norm(np.array(self.targets[i].state[:2]) - np.array(self.agent.state[:2])))
        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training,
                                                                        is_col=is_col, observed=observed)
        # reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(self.is_training,
        #                                                                 is_col=is_col)
        # Compute the RL state.
        self.state_func(action_vw, observed)

        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        return self.state, reward, False, False, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = {}
        particle_list = []
        observed_list = []
        for i in range(self.num_targets):
            # r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
            #                                             xy_base=self.agent.state[:2],
            #                                             theta_base=self.agent.state[2])
            # r_dot_b, alpha_dot_b = util.relative_velocity_polar(
            #     self.belief_targets[i].state[:2],
            #     self.belief_targets[i].state[2:],
            #     self.agent.state[:2], self.agent.state[2],
            #     action_vw[0], action_vw[1])
            particle_list += [[x] + list(util.relative_distance_polar(y[:2],
                                                                      xy_base=self.agent.state[:2],
                                                                      theta_base=self.agent.state[2])) + list(
                util.relative_velocity_polar(
                    y[:2],
                    y[2:],
                    self.agent.state[:2], self.agent.state[2],
                    action_vw[0], action_vw[1])) for x, y in
                              zip(self.belief_targets[i].weights, self.belief_targets[i].states)]

            # particle_list += [np.concatenate(([x],y),axis=0) for x, y in
            #                   zip(self.belief_targets[i].weights, self.belief_targets[i].states)]

            #
            # observed_list = np.concatenate((observed_list, [LA.det(self.belief_targets[i].cov)]))
            # observed_list = np.concatenate((observed_list, [float(observed[i])]))

        observed_list = np.concatenate((observed_list, obstacles_pt))
        observed_list = np.concatenate((observed_list, self.agent.state))
        self.state = {"target": torch.tensor(np.array(particle_list), dtype=torch.float32, device=DEVICE).unsqueeze(0),
                      "agent": torch.tensor(np.array([observed_list]), dtype=torch.float32, device=DEVICE).unsqueeze(0)}

        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def observe_and_update_belief(self):
        observed = []
        for i in range(self.num_targets):
            observation = self.observation(self.targets[i])
            observed.append(observation[0])
            if observation[0]:  # if observed, update the target belief.
                self.belief_targets[i].update(observation, self.agent.state)
                self.discover_cnt[i] += 1
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
        return observed

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  # Maximum relative speed

        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(
            ([0.0, -np.pi, -rel_speed_limit, -10 * np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
            np.concatenate((
                [600.0, np.pi, rel_speed_limit, 10 * np.pi, 50.0, 2.0] * self.num_targets,
                [self.sensor_r, np.pi]))]
        # self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        self.observation_space = spaces.Box(self.limit['state'][0][:4], self.limit['state'][1][:4], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def _generate_prior_dist(self, init_pose):
        # var = 1.6
        # # init_pose = super().reset()
        # dists = []
        # for i in range(self.num_targets):
        #     # init_mean = init_pose['belief_targets'][i][:self.target_dim]
        #     x_size = [self.MAP.mapmin[0], self.MAP.mapmax[0]]
        #     y_size = [self.MAP.mapmin[1], self.MAP.mapmax[1]]
        #     # x_size = [init_mean[0]-np.sqrt(self.target_init_cov), self.MAP.mapmax[0]+np.sqrt(self.target_init_cov)]
        #     # y_size = [self.MAP.mapmin[1]-np.sqrt(self.target_init_cov), self.MAP.mapmax[1]+np.sqrt(self.target_init_cov)]
        #     n_bins = int(self.MAP.mapmax[0] - self.MAP.mapmin[0])
        #     dx = dy = np.linspace(-self.target_init_cov,self.target_init_cov,n_bins)
        #     x = np.linspace(x_size[0], x_size[1], n_bins)
        #     y = np.linspace(y_size[0], y_size[1], n_bins)
        #     xx, yy, xy, yx = np.meshgrid(x, y, dx, dy)
        #     positions = np.vstack([xx.ravel(), yy.ravel(), xy.ravel(), yx.ravel()]).T
        #     prior_gmm = GMMDist([1 / (n_bins ** self.target_dim)] * (n_bins ** self.target_dim), positions,
        #                         np.array([np.diag([var] * self.target_dim)] * (n_bins ** self.target_dim)))
        #     dists.append(prior_gmm)
        # self.prior_dists = dists
        # return dists
        dists = [GaussianDistribution(
            np.concatenate((init_pose['belief_targets'][i][:2], np.ones(2) * (METADATA['target_speed_limit'] / 2))),
            np.eye(self.target_dim) * self.target_init_cov) for i in range(self.num_targets)]
        return dists

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                              collision_func=lambda x: self.MAP.is_collision(x))
        # Build targets
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                                       self.sampling_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.sampling_period ** 3 / 3 * np.eye(2),
                            self.sampling_period ** 2 / 2 * np.eye(2)), axis=1),
            np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                            self.sampling_period * np.eye(2)), axis=1)))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * np.concatenate((
                np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                                self.sampling_period / 2 * np.eye(2)), axis=1),
                np.concatenate((self.sampling_period / 2 * np.eye(2),
                                self.sampling_period * np.eye(2)), axis=1)))

        self.target_transition = LinearGaussianDistribution(coefficient=self.targetA,
                                                            noise_mean=np.zeros(self.target_dim),
                                                            noise_var=self.target_true_noise_sd)
        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                                                   self.sampling_period, self.limit['target'],
                                                   lambda x: self.MAP.is_collision(x),
                                                   W=self.target_true_noise_sd, A=self.targetA,
                                                   obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                                       x, fov=2 * np.pi, r_max=10e2))
                        for _ in range(self.num_targets)]

        self.belief_targets = [
            PFbelief(dim=self.target_dim, limit=self.limit['target'], transition_func=self.target_transition,
                     n=self.n_particles, effective_n=self.n_particles // 2, dim_z=2,
                     obs_noise_func=self.observation_noise,
                     collision_func=lambda x: self.MAP.is_collision(x),
                     obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                         x, fov=2 * np.pi, r_max=10e2), sampling_period=self.sampling_period)
            for _ in range(self.num_targets)]



        # self.belief_targets = [
        #     AuxiliaryPFbelief(dim=self.target_dim, limit=self.limit['target'], transition_func=self.target_transition,
        #                       n=self.n_particles, effective_n=self.n_particles // 2, dim_z=2,
        #                       obs_noise_func=self.observation_noise,
        #                       collision_func=lambda x: self.MAP.is_collision(x),
        #                       obs_check_func=lambda x: self.MAP.get_closest_obstacle(
        #                           x, fov=2 * np.pi, r_max=10e2), sampling_period=self.sampling_period)
        #     for _ in range(self.num_targets)]

    def get_reward(self, is_training=True, **kwargs):
        # reward = np.sum([1 / (np.linalg.norm(
        #     np.array(bf.state[:2]) - np.array(self.agent.state[:2])) + 0.001) for i, bf in
        #                  enumerate(self.belief_targets)])
        # return reward, False, 0, 0

        # reward = np.sum([self.last_ents[i] - bf.entropy() for i, bf in
        #                  enumerate(self.belief_targets)])
        # return reward, False, 0, 0
        # if self.observation_model == LEAKAGE_OBS:
        #     reward = np.sum([self.last_ents[i] - bf.entropy() for i, bf in
        #                      enumerate(self.belief_targets)])
        #     # ob_reward = np.sum([self.observation(target)[1] for target in self.targets])
        #     # reward += ob_reward
        #     detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        #     r_detcov_mean = - np.mean(np.log(detcov))
        #     r_detcov_std = - np.std(np.log(detcov))
        #     return reward, False, r_detcov_mean, r_detcov_std
        # else:
        #     # c_penalty = 1.0
        #     # mis = [self.last_ents[i] - b_target.entropy() for i, b_target in enumerate(self.belief_targets)]
        #     # detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        #     # r_detcov_mean = - np.mean(np.log(detcov))
        #     # r_detcov_std = - np.std(np.log(detcov))
        #     # normed_ent_reward = np.mean(mis) / self.max_ent
        #     # ob_reward = np.sum([float(ob) for ob in kwargs["observed"]])
        #     # reward = normed_ent_reward + ob_reward
        #     # # reward = np.sum([float(ob) for ob in kwargs["observed"]])
        #     # if "is_col" in kwargs.keys() and kwargs["is_col"]:z
        #     #     reward = reward - 1.0 * c_penalty
        #     # return reward, False, r_detcov_mean,r_detcov_std
        #     mis = [self.last_ents[i] - b_target.entropy() for i, b_target in enumerate(self.belief_targets)]
        #     # mis = [ - b_target.entropy() for i, b_target in enumerate(self.belief_targets)]
        #     detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        #     r_detcov_mean = - np.mean(np.log(detcov))
        #     r_detcov_std = - np.std(np.log(detcov))
        #     reward = np.sum(mis)
        #     return reward, False, r_detcov_mean, r_detcov_std
        reward = np.sum([self.last_ents[i] - bf.entropy() for i, bf in
                         enumerate(self.belief_targets)])
        # ob_reward = np.sum([self.observation(target)[1] for target in self.targets])
        # reward += ob_reward
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))
        c_penalty = 1.0
        if "is_col" in kwargs.keys() and kwargs["is_col"]:
            reward = reward - 1.0 * c_penalty
        return reward, False, r_detcov_mean, r_detcov_std

        # xy_target_base = [util.transform_2d(bs.state[:2], self.agent.state[2], self.agent.state[:2]) for bs in
        #                   self.belief_targets]
        # min_dist_to_perfect = .05
        # m_reward = 1.0 / min_dist_to_perfect
        # reward = np.sum(
        #         [m_reward if kwargs["observed"][i] else self.last_ents[i] - self.belief_targets[i].entropy() / np.clip(LA.norm(np.array(xy_target_base[i]) - np.array([5, 0])), min_dist_to_perfect, np.inf)
        #          for i in range(len(xy_target_base))])

    #     c_mean = 0.1
    #     c_std = 0.0
    #     c_penalty = 1.0
    #     detcov = np.array([LA.det(b_target.cov) for b_target in self.belief_targets])
    #     detcov[np.where(detcov <= 0)] = 10 ** -9
    #     r_detcov_mean = - np.mean(np.log(detcov))
    #     r_detcov_std = - np.std(np.log(detcov))
    #
    #     reward = c_mean * r_detcov_mean + c_std * r_detcov_std
    #     reward = min(0.0, reward) - c_penalty * 1.0
    #
    #     return reward, False, r_detcov_mean, r_detcov_std

    # def get_reward(self, is_training=True, **kwargs):
    #     reward = np.sum([float(ob) for ob in kwargs["observed"]])
    # #     xy_target_base = [util.transform_2d(bs.state[:2], self.agent.state[2], self.agent.state[:2]) for bs in
    # #                       self.belief_targets]
    # #     min_dist_to_perfect = .05
    # #     m_reward = 1.0 / min_dist_to_perfect
    # #     reward = np.sum(
    # #         [m_reward if ob else 1 / np.clip(LA.norm(np.array(xy) - np.array([5, 0])), min_dist_to_perfect, np.inf) for
    # #          ob, xy in
    # #          zip(kwargs["observed"], xy_target_base)])
    #     return reward, False, 0,0
    # reward = np.sum(
    #     [-LA.norm(np.array(bs.state[:2]) - np.array(self.agent.state[:2])) for bs in self.targets])
    # return reward, False, 0, 0
    # if "observed" in kwargs:
    #     max_d = np.sqrt(
    #         (self.MAP.mapmax[0] - self.MAP.mapmin[0]) ** 2 + (self.MAP.mapmax[1] - self.MAP.mapmin[1]) ** 2)
    #     reward = np.sum([1.0 - (np.linalg.norm(
    #         np.array(self.belief_targets[i].state[:2]) - np.array(self.agent.state[:2]))) / max_d if not ob else 1.0
    #                      for i, ob in enumerate(kwargs["observed"])])
    # else:
    #     reward = np.sum(
    #         [self.last_est_dists[i] - np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for
    #          i, bf in enumerate(self.belief_targets)])
    # reward = np.sum([(self.last_ents[i] - bf.entropy()) / np.linalg.norm(
    #     np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for i, bf in
    #                  enumerate(self.belief_targets)])
    # reward = np.sum(
    #     [self.last_est_dists[i] - np.linalg.norm(np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for i, bf
    #      in enumerate(self.belief_targets)])
    # reward = np.sum([(self.last_ents[i] - bf.entropy()) / np.linalg.norm(
    #     np.array(bf.state[:2]) - np.array(self.agent.state[:2])) for i, bf in
    #                  enumerate(self.belief_targets)])
    # reward = np.sum([-np.linalg.norm(np.array(self.belief_targets[i].state[:2]) - np.array(self.agent.state[:2])) if not ob else 100.0 for i, ob in enumerate(kwargs["observed"])])
    # return reward, False, 0,0
    # observability = []
    # for bs in self.belief_targets:
    #     ob_cnt = 0.0
    #     unob_cnt = 0.0
    #     for w,p in zip(bs.weights,bs.states):
    #         r,alpha = util.relative_distance_polar(p[:2],
    #                                      xy_base=self.agent.state[:2],
    #                                      theta_base=self.agent.state[2])
    #         observed = (r <= self.sensor_r) \
    #                    & (abs(alpha) <= self.fov / 2 / 180 * np.pi) \
    #                    & (not (self.MAP.is_blocked(self.agent.state, p)))
    #         if observed:
    #             ob_cnt +=w
    #         else:
    #             unob_cnt+=w
    #
    #     observability.append((ob_cnt,unob_cnt))
    # r, alpha = util.relative_distance_polar(bs.state[:2],
    #                                         xy_base=self.agent.state[:2],
    #                                         theta_base=self.agent.state[2])
    # observed = (r <= self.sensor_r) \
    #            & (abs(alpha) <= self.fov / 2 / 180 * np.pi) \
    #            & (not (self.MAP.is_blocked(self.agent.state, bs.state)))
    # if observed:
    #     observability.append(1)
    # else:
    #     observability.append(0)
    # c_mean = 0.1
    # c_std = 0.0
    # c_penalty = 1.0

    # detcov = np.array([LA.det(bs.cov)/LA.det(self.observation_noise(())) for bs in self.belief_targets])
    # detcov[np.where(detcov <= 0)] = 10 ** -9
    # r_detcov_mean = - np.mean(np.log(detcov))
    # r_detcov_std = - np.std(np.log(detcov))
    #
    # reward = c_mean * r_detcov_mean + c_std * r_detcov_std
    # reward = min(0.0, reward) - c_penalty * 1.0

    # return reward, False, r_detcov_mean, r_detcov_std
    # reward = np.sum([1-ob[1] for ob in observability])
    # return reward, False, 0, 0


class TargetTrackingEnv2(TargetTrackingEnv0):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
                                    map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v2'
        self.target_dim = 3

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 4
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit

        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
                               np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                              lambda x: self.MAP.is_collision(x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period,
                                 self.limit['target'],
                                 lambda x: self.MAP.is_collision(x),
                                 policy=SinePolicy(0.1, 0.5, 5.0, self.sampling_period))
                        for _ in range(self.num_targets)]

        self.target_noise_cov = self.const_q * self.sampling_period * np.eye(self.target_dim)
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * \
                                        self.sampling_period * np.eye(self.target_dim)
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim,
                                         limit=self.limit['target'], fx=SE2Dynamics,
                                         W=self.target_noise_cov,
                                         obs_noise_func=self.observation_noise,
                                         collision_func=lambda x: self.MAP.is_collision(x))
                               for _ in range(self.num_targets)]


class TargetTrackingEnv3(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingEnv0.__init__(self, num_targets=num_targets,
                                    map_name=map_name, is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'TargetTracking-v3'
        self.target_dim = 5
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits()

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def set_limits(self, target_speed_limit=None):
        if self.target_dim != 5:
            return
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  # Maximum relative speed

        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  # Maximum relative speed
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-np.pi, -self.target_speed_limit, -np.pi])),
                                np.concatenate((self.MAP.mapmax, [np.pi, self.target_speed_limit, np.pi]))]
        self.limit['state'] = [np.concatenate(
            ([0.0, -np.pi, -rel_speed_limit, -10 * np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
            np.concatenate((
                [600.0, np.pi, rel_speed_limit, 10 * np.pi, 50.0, 2.0] * self.num_targets,
                [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if self.target_dim != 5:
            return

        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        self.target_noise_cov = np.zeros((self.target_dim, self.target_dim))
        for i in range(3):
            self.target_noise_cov[i, i] = self.const_q * self.sampling_period ** 3 / 3
        self.target_noise_cov[3:, 3:] = self.const_q * \
                                        np.array([[self.sampling_period, self.sampling_period ** 2 / 2],
                                                  [self.sampling_period ** 2 / 2, self.sampling_period]])
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * \
                                        self.sampling_period * np.eye(self.target_dim)
        # Build a robot
        self.agent = AgentSE2(3, self.sampling_period, self.limit['agent'],
                              lambda x: self.MAP.is_collision(x))
        # Build a target
        self.targets = [AgentSE2(self.target_dim, self.sampling_period, self.limit['target'],
                                 lambda x: self.MAP.is_collision(x),
                                 policy=ConstantPolicy(self.target_noise_cov[3:, 3:]))
                        for _ in range(self.num_targets)]
        # SinePolicy(0.5, 0.5, 2.0, self.sampling_period)
        # CirclePolicy(self.sampling_period, self.MAP.origin, 3.0)
        # RandomPolicy()

        self.belief_targets = [UKFbelief(dim=self.target_dim,
                                         limit=self.limit['target'], fx=SE2DynamicsVel,
                                         W=self.target_noise_cov,
                                         obs_noise_func=self.observation_noise,
                                         collision_func=lambda x: self.MAP.is_collision(x))
                               for _ in range(self.num_targets)]

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        self.discover_cnt = [0 for _ in range(self.num_targets)]
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                init_state=np.concatenate((init_pose['belief_targets'][i], np.zeros(2))),
                init_cov=self.target_init_cov)
            t_init = np.concatenate((init_pose['targets'][i], [self.target_init_vel[0], 0.0]))
            self.targets[i].reset(t_init)
            self.targets[i].policy.reset(t_init)

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                        xy_base=self.agent.state[:2],
                                                        theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar_se2(
                self.belief_targets[i].state[:3],
                self.belief_targets[i].state[3:],
                self.agent.state, action_vw)
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                               np.log(LA.det(self.belief_targets[i].cov)), float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)
        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def observe_and_update_belief(self):
        observed = []
        for i in range(self.num_targets):
            observation = self.observation(self.targets[i])
            observed.append(observation[0])
            if observation[0]:  # if observed, update the target belief.
                self.belief_targets[i].update(observation, self.agent.state)
                self.discover_cnt[i] += 1
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
        return observed
