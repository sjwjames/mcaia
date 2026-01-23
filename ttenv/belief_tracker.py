"""Belief Trackers

KFbelief : Belief Update using Kalman Filter
UKFbelief : Belief Update using Unscented Kalman Filter using filterpy library
"""
import numpy as np
import torch
from numpy import linalg as LA
from scipy.stats import multivariate_normal

import ttenv.util as util

from filterpy.kalman import JulierSigmaPoints, UnscentedKalmanFilter, ExtendedKalmanFilter

from ttenv.base_model import ParticleDist, LinearGaussianDistribution, GMMDist, batch_mvnorm_logpdf, \
    batch_mvnorm_logpdf_multi_cov, GasLeakageModel
from ttenv.metadata import METADATA, GAUSSIAN_OBS, LEAKAGE_OBS

REG_PARAM = 1e-9


class KFbelief(object):
    """
    Kalman Filter for the target tracking problem.

    state : target state
    x : agent state
    z : observation (r, alpha)
    """

    def __init__(self, dim, limit, dim_z=2, A=None, W=None,
                 obs_noise_func=None, collision_func=None):
        """
        dim : dimension of state
        limit : An array of two vectors.
                limit[0] = minimum values for the state,
                limit[1] = maximum value for the state
        dim_z : dimension of observation,
        A : state transition matrix
        W : state noise matrix
        obs_noise_func : observation noise matrix function of z
        collision_func : collision checking function
        """
        self.dim = dim
        self.limit = limit
        self.A = np.eye(self.dim) if A is None else A
        self.W = W if W is not None else np.zeros((self.dim, self.dim))
        self.obs_noise_func = obs_noise_func
        self.collision_func = collision_func

    def reset(self, init_state, init_cov):
        self.state = init_state
        self.cov = init_cov * np.eye(self.dim)

    def predict(self):
        # Prediction
        state_new = np.matmul(self.A, self.state)
        self.cov = np.matmul(np.matmul(self.A, self.cov), self.A.T) + self.W
        self.state = np.clip(state_new, self.limit[0], self.limit[1])

    def update(self, z_t, x_t):
        """
        Parameters
        --------
        z_t : observation - radial and angular distances from the agent.
        x_t : agent state (x, y, orientation) in the global frame.
        """
        # Kalman Filter Update
        r_pred, alpha_pred = util.relative_distance_polar(
            self.state[:2], x_t[:2], x_t[2])
        diff_pred = np.array(self.state[:2]) - np.array(x_t[:2])
        if self.dim == 2:
            Hmat = np.array([[diff_pred[0], diff_pred[1]],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred]]) / r_pred
        elif self.dim == 4:
            Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
        else:
            raise ValueError('target dimension for KF must be either 2 or 4')
        innov = z_t - np.array([r_pred, alpha_pred])
        innov[1] = util.wrap_around(innov[1])

        R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
            + self.obs_noise_func((r_pred, alpha_pred))
        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)

        self.cov = np.matmul(C, self.cov)
        self.state = np.clip(self.state + np.matmul(K, innov), self.limit[0], self.limit[1])

    def entropy(self):
        k = 2
        return 0.5 * np.log(np.linalg.det(self.cov)) + 0.5 * np.log(2 * np.pi) * k + 0.5 * k


class UKFbelief(object):
    """
    Unscented Kalman Filter from filterpy
    """

    def __init__(self, dim, limit, dim_z=2, fx=None, W=None, obs_noise_func=None,
                 collision_func=None, sampling_period=0.5, kappa=1, hx=None, measurement_model=GAUSSIAN_OBS):
        """
        dim : dimension of state
            ***Assuming dim==3: (x,y,theta), dim==4: (x,y,xdot,ydot), dim==5: (x,y,theta,v,w)
        limit : An array of two vectors. limit[0] = minimum values for the state,
                                            limit[1] = maximum value for the state
        dim_z : dimension of observation,
        fx : x_tp1 = fx(x_t, dt), state dynamic function
        W : state noise matrix
        obs_noise_func : observation noise matrix function of z
        collision_func : collision checking function
        n : the number of sigma points
        """
        self.dim = dim
        self.limit = limit
        self.W = W if W is not None else np.zeros((self.dim, self.dim))
        self.obs_noise_func = obs_noise_func
        self.collision_func = collision_func
        self.measurement_model = measurement_model
        if hx is None:
            def hx(y, agent_state, measure_func=util.relative_distance_polar):
                r_pred, alpha_pred = measure_func(y[:2], agent_state[:2],
                                                  agent_state[2])
                return np.array([r_pred, alpha_pred])

        def x_mean_fn_(sigmas, Wm):
            if dim == 3:
                x = np.zeros(dim)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += np.sin(s[2]) * Wm[i]
                    sum_cos += np.cos(s[2]) * Wm[i]
                x[2] = np.arctan2(sum_sin, sum_cos)
                return x
            elif dim == 5:
                x = np.zeros(dim)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    x[3] += s[3] * Wm[i]
                    x[4] += s[4] * Wm[i]
                    sum_sin += np.sin(s[2]) * Wm[i]
                    sum_cos += np.cos(s[2]) * Wm[i]
                x[2] = np.arctan2(sum_sin, sum_cos)
                return x
            else:
                return None

        def z_mean_fn_(sigmas, Wm):
            x = np.zeros(dim_z)
            sum_sin, sum_cos = 0., 0.
            for i in range(len(sigmas)):
                s = sigmas[i]
                x[0] += s[0] * Wm[i]
                sum_sin += np.sin(s[1]) * Wm[i]
                sum_cos += np.cos(s[1]) * Wm[i]
            x[1] = np.arctan2(sum_sin, sum_cos)
            return x

        def residual_x_(x, xp):
            """
            x : state, [x, y, theta]
            xp : predicted state
            """
            if dim == 3 or dim == 5:
                r_x = x - xp
                r_x[2] = util.wrap_around(r_x[2])
                return r_x
            else:
                return None

        def residual_z_(z, zp):
            """
            z : observation, [r, alpha]
            zp : predicted observation
            """
            r_z = z - zp
            r_z[1] = util.wrap_around(r_z[1])
            return r_z

        sigmas = JulierSigmaPoints(n=dim, kappa=kappa)
        if measurement_model == GAUSSIAN_OBS:
            self.ukf = UnscentedKalmanFilter(dim, dim_z, sampling_period, fx=fx,
                                             hx=hx, points=sigmas, x_mean_fn=x_mean_fn_,
                                             z_mean_fn=z_mean_fn_, residual_x=residual_x_,
                                             residual_z=residual_z_)
        elif measurement_model == LEAKAGE_OBS:
            self.ukf = UnscentedKalmanFilter(dim, dim_z, sampling_period, fx=fx,
                                             hx=hx, points=sigmas)

    def reset(self, init_state, init_cov):
        self.state = init_state
        self.cov = init_cov * np.eye(self.dim)
        self.ukf.x = self.state
        self.ukf.P = self.cov
        self.ukf.Q = self.W  # process noise matrix

    def predict(self, u_t=None):
        if u_t is None:
            u_t = np.array([np.random.random(),
                            np.pi * np.random.random() - 0.5 * np.pi])

        # Kalman Filter Update
        if self.measurement_model == LEAKAGE_OBS:
            self.ukf.predict()
        else:
            self.ukf.predict(u=u_t)
        self.cov = self.ukf.P
        self.state = np.clip(self.ukf.x, self.limit[0], self.limit[1])

    def update(self, z_t, x_t):
        # Kalman Filter Update

        if self.measurement_model == LEAKAGE_OBS:
            self.ukf.update(z_t, R=np.array([0.01]), agent_coord=x_t)
        else:
            r_pred, alpha_pred = util.relative_distance_polar(self.ukf.x[:2], x_t[:2], x_t[2])
            self.ukf.update(z_t, R=self.obs_noise_func((r_pred, alpha_pred)),
                            agent_state=x_t)

        self.cov = self.ukf.P
        self.state = np.clip(self.ukf.x, self.limit[0], self.limit[1])

    def entropy(self):
        k = 2
        return 0.5 * np.log(np.linalg.det(self.cov)) + 0.5 * np.log(2 * np.pi) * k + 0.5 * k


class PFbelief(object):
    def __init__(self, dim, limit, transition_func, n, effective_n, dim_z=2,
                 obs_noise_func=None, collision_func=None, obs_check_func=None, sampling_period=0.5):
        self.dim = dim
        self.dim_z = dim_z
        self.limit = limit
        self.n = n
        self.effective_n = effective_n
        self.weights = np.array([1 / n] * n)
        self.transition_func = transition_func
        self.collision_func = collision_func
        self.obs_noise_func = obs_noise_func
        self.obs_check_func = obs_check_func
        self.sampling_period = sampling_period
        if METADATA["observation_model"] == LEAKAGE_OBS:
            self.leakage_model = GasLeakageModel()

    def reset(self, prior_dist):
        self.states = prior_dist.sample(self.n)
        # if self.collision_func(np.dot(self.weights, self.states)):
        #     self.states = [self.collision_deviation(p) for p in self.states]
        # self.states = np.concatenate((np.clip(self.states[:,:2], self.limit[0][:2], self.limit[1][:2]),self.states[:,2:]),axis=1)
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.weights = np.array([1 / self.n] * self.n)
        self.state, self.cov = self.calculate_bs_moments()

    def predict(self):
        self.states = self.transition_func.sample(self.states, self.n)
        # self.states = np.array([np.matmul(self.transition_func.coefficient,state) for state in self.states])
        # if self.collision_func(np.dot(self.weights, self.states)):
        #     self.states = [self.collision_deviation(p) for p in self.states]
        # self.states = np.concatenate((np.clip(self.states[:,:2], self.limit[0][:2], self.limit[1][:2]),self.states[:,2:]),axis=1)
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.state, self.cov = self.calculate_bs_moments()

    def generate_predictive_samples(self):
        states = self.transition_func.sample(self.states, self.n)
        states = np.clip(states, self.limit[0], self.limit[1])
        return states

    def sample_next_belief(self, observation_info, next_state_samples, agent_state):
        observed = observation_info[0]
        if not observed:
            return self.weights, next_state_samples
        observation = observation_info[1]
        if METADATA["observation_model"] == GAUSSIAN_OBS:
            un_normed_cond_ll_logprobs = [
                multivariate_normal.logpdf(observation, list(util.relative_distance_polar(state_sample[:2],
                                                                                          xy_base=agent_state[:2],
                                                                                          theta_base=agent_state[2])),
                                           self.obs_noise_func(observation), allow_singular=True)
                for state_sample in
                next_state_samples]
        else:
            un_normed_cond_ll_logprobs = self.leakage_model.measurement_log_llhs(agent_state, next_state_samples,
                                                                                 observation)
        log_weights_new = un_normed_cond_ll_logprobs + np.log(self.weights)
        log_weights_new = log_weights_new - np.max(log_weights_new)
        if np.sum(np.exp(log_weights_new)) != 0:
            weights_new = np.exp(log_weights_new) / np.sum(np.exp(log_weights_new))
        else:
            weights_new = self.weights

        sample_efficiency = 1 / np.sum(weights_new ** 2)
        if sample_efficiency < self.effective_n:
            random_samples = np.random.choice(next_state_samples.shape[0], size=self.n, replace=True, p=weights_new)
            next_state_samples = next_state_samples[random_samples]
            weights_new = np.ones(self.n) * (1 / self.n)
        next_state_samples = np.clip(next_state_samples, self.limit[0], self.limit[1])
        return weights_new, next_state_samples

    def update(self, observation_info, agent_state):
        observed = observation_info[0]
        observation = observation_info[1]
        next_state_samples = self.states
        # obs_means = [list(util.relative_distance_polar(state_sample[:2],
        #                                                xy_base=agent_state[:2],
        #                                                theta_base=agent_state[2])) for state_sample in
        #              next_state_samples]
        # obs_noise_covs = [self.obs_noise_func(observation, target=state_sample[:2], agent=agent_state[:2]) for
        #                   state_sample in
        #                   next_state_samples]
        # un_normed_cond_ll_logprobs = batch_mvnorm_logpdf(np.array(observation),np.array(obs_means),np.array(obs_noise_covs))
        # un_normed_cond_ll_logprobs = batch_mvnorm_logpdf_multi_cov(np.array(observation), np.array(obs_means),
        #                                                            np.array(obs_noise_covs))
        if METADATA["observation_model"] == GAUSSIAN_OBS:
            un_normed_cond_ll_logprobs = [
                multivariate_normal.logpdf(observation, list(util.relative_distance_polar(state_sample[:2],
                                                                                          xy_base=agent_state[:2],
                                                                                          theta_base=agent_state[2])),
                                           self.obs_noise_func(observation), allow_singular=True)
                for state_sample in
                next_state_samples]
        else:
            un_normed_cond_ll_logprobs = self.leakage_model.measurement_log_llhs(agent_state, next_state_samples,
                                                                                 observation)
        log_weights_new = un_normed_cond_ll_logprobs + np.log(self.weights)
        log_weights_new = log_weights_new - np.max(log_weights_new)
        if np.sum(np.exp(log_weights_new)) != 0:
            weights_new = np.exp(log_weights_new) / np.sum(np.exp(log_weights_new))
        else:
            weights_new = self.weights

        sample_efficiency = 1 / np.sum(weights_new ** 2)
        if sample_efficiency < self.effective_n:
            random_samples = np.random.choice(next_state_samples.shape[0], size=self.n, replace=True, p=weights_new)
            next_state_samples = next_state_samples[random_samples]
            weights_new = np.ones(self.n) * (1 / self.n)
            # post_dist = GMMDist(weights_new, pred_state_marg.means, pred_state_marg.covs)
            # sampled_data = post_dist.sample(N)

        self.weights = weights_new
        self.states = next_state_samples
        # if self.collision_func(np.dot(weights_new, next_state_samples)):
        #     self.states = [self.collision_deviation(p) for p in next_state_samples]
        # self.states = np.concatenate((np.clip(self.states[:,:2], self.limit[0][:2], self.limit[1][:2]),self.states[:,2:]),axis=1)
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.state, self.cov = self.calculate_bs_moments()

    def resample(self):
        if np.ndim(self.states) > 1:
            random_samples = np.random.choice(np.array(self.states).shape[0], size=self.n, replace=True, p=self.weights)
            bs_resampled = self.states[random_samples]
        else:
            bs_resampled = np.random.choice(self.states, self.n, replace=True, p=self.weights)
        return bs_resampled


    @staticmethod
    def calculate_bs_moments_static(weights, states):
        GMMDist(weights, states)

    def calculate_bs_moments(self):
        # mean_val = np.dot(self.weights, self.states)
        state_dim = len(self.states[0])
        gmm_approx = GMMDist(self.weights, self.states, [np.eye(state_dim) * 0.01 for _ in self.weights])
        mean_val, var_val = gmm_approx.compute_mms()
        # if len(mean_val) > 1:
        #     mean_val = np.average(self.states, axis=0, weights=self.weights)
        #     var_val = np.cov(self.states, rowvar=False, aweights=self.weights)
        #
        #     # deviation = np.array(self.states) - mean_val
        #     # correlations = [np.outer(deviation[i], deviation[i]) for i in range(self.n)]
        #     # var_val = np.sum([correlations[i] * self.weights[i] for i in range(self.n)], axis=0)
        # else:
        #     var_val = np.dot(self.weights, (np.array(self.states) - mean_val) ** 2)
        return mean_val, var_val

    # @staticmethod
    # def entropy_static(states, weights):
    #     state_dim = len(states[0])
    #     gmm_approx = GMMDist(weights, states, [np.eye(state_dim) * 0.01 for _ in weights])
    #     ent = gmm_approx.sg_entropy_ub()
    #     return ent

    def entropy(self):
        state_dim = len(self.states[0])
        gmm_approx = GMMDist(self.weights, self.states, [np.eye(state_dim) * 0.01 for _ in self.weights])
        ent = gmm_approx.sg_entropy_ub()
        return ent
        # m, c = self.calculate_bs_moments()
        # k = len(m)
        # return 0.5 * np.log(np.linalg.det(c)) + 0.5 * np.log(2 * np.pi) * k + 0.5 * k
        # agg = defaultdict(float)
        # for pt, w in zip(self.states, self.weights):
        #     agg[tuple(pt)] += w
        # agg_weights = list(agg.values())
        # return -np.dot(agg_weights, np.log(agg_weights))

    def collision_deviation(self, state):
        new_state = self.collision_control(state)
        if self.obs_check_func is not None:
            del_vx, del_vy = self.obstacle_detour_maneuver(new_state,
                                                           r_margin=METADATA[
                                                                        'target_speed_limit'] * self.sampling_period * 2)
            new_state[2] += del_vx
            new_state[3] += del_vy
        return new_state

    def collision_control(self, state):
        """
        Assigns a new velocity deviating the agent with an angle (pi/2, pi) from
        the closest obstacle point.
        """
        odom = [state[0], state[1], np.arctan2(state[3], state[2])]
        obs_pos = self.obs_check_func(odom)
        v = np.sqrt(np.sum(np.square(state[2:]))) + np.random.normal(0.0, 1.0)
        if obs_pos[1] >= 0:
            th = obs_pos[1] - (1 + np.random.random()) * np.pi / 2
        else:
            th = obs_pos[1] + (1 + np.random.random()) * np.pi / 2

        state_new = np.array([state[0], state[1], v * np.cos(th + odom[2]), v * np.sin(th + odom[2])])
        return state_new

    def obstacle_detour_maneuver(self, state, r_margin=1.0):
        """
        Returns del_vx, del_vy which will be added to the new state.
        This provides a repultive force from the closest obstacle point based
        on the current velocity, a linear distance, and an angular distance.

        Parameters:
        ----------
        r_margin : float. A margin from an obstalce that you want to consider
        as the minimum distance the target can get close to the obstacle.
        """
        odom = [state[0], state[1], np.arctan2(state[3], state[2])]
        obs_pos = self.obs_check_func(odom)
        speed = np.sqrt(np.sum(state[2:] ** 2))
        rot_ang = np.pi / 2 * (1. + 1. / (1. + np.exp(-(speed - 0.5 * METADATA['target_speed_limit']))))
        if obs_pos is not None:
            acc = max(0.0, speed * np.cos(obs_pos[1])) / max(METADATA['margin2wall'], obs_pos[0] - r_margin)
            th = obs_pos[1] - rot_ang if obs_pos[1] >= 0 else obs_pos[1] + rot_ang
            del_vx = acc * np.cos(th + odom[2]) * self.sampling_period
            del_vy = acc * np.sin(th + odom[2]) * self.sampling_period
            return del_vx, del_vy
        else:
            return 0., 0.


class AuxiliaryPFbelief(PFbelief):
    def __init__(self, dim, limit, transition_func, n, effective_n, dim_z=2,
                 obs_noise_func=None, collision_func=None, obs_check_func=None, sampling_period=0.5):
        super().__init__(dim, limit, transition_func, n, effective_n, dim_z,
                         obs_noise_func, collision_func, obs_check_func, sampling_period)
        self.last_states = None
        self.last_weights = None

    def predict(self):
        self.last_states = np.array(self.states)
        self.last_weights = np.array(self.weights)
        self.states = self.transition_func.sample(self.states, self.n)
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.state, self.cov = self.calculate_bs_moments()

    def reset(self, prior_dist):
        self.states = prior_dist.sample(self.n)
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.weights = np.array([1 / self.n] * self.n)
        self.last_states = np.array(self.states)
        self.last_weights = np.array(self.weights)
        self.state, self.cov = self.calculate_bs_moments()

    def update(self, observation_info, agent_state):
        observed = observation_info[0]
        observation = observation_info[1]
        last_state_samples_indices = np.random.choice(self.last_states.shape[0], size=self.n, replace=True,
                                                      p=self.last_weights)
        last_state_samples = self.last_states[last_state_samples_indices]
        mus = self.transition_func.compute_marginal_mean(last_state_samples)
        un_normed_cond_ll_logprobs = [
            multivariate_normal.logpdf(observation, list(util.relative_distance_polar(state_sample[:2],
                                                                                      xy_base=agent_state[:2],
                                                                                      theta_base=agent_state[2])),
                                       self.obs_noise_func(observation), allow_singular=True)
            for state_sample in
            mus]
        # log_weights_new = un_normed_cond_ll_logprobs + np.log(self.weights)
        un_normed_cond_ll_logprobs = np.array(un_normed_cond_ll_logprobs) - np.max(un_normed_cond_ll_logprobs)
        first_stage_weights = np.exp(un_normed_cond_ll_logprobs) / np.sum(np.exp(un_normed_cond_ll_logprobs))
        preselected_samples_indices = np.random.choice(self.last_states.shape[0], size=self.n, replace=True,
                                                       p=first_stage_weights)
        preselected_samples = self.last_states[preselected_samples_indices]
        next_state_samples = self.transition_func.sample(preselected_samples, self.n)
        un_normed_cond_ll_logprobs_new = [
            multivariate_normal.logpdf(observation, list(util.relative_distance_polar(state_sample[:2],
                                                                                      xy_base=agent_state[:2],
                                                                                      theta_base=agent_state[2])),
                                       self.obs_noise_func(observation), allow_singular=True)
            for state_sample in
            next_state_samples]
        un_normed_cond_ll_logprobs_new = np.array(un_normed_cond_ll_logprobs_new) - np.max(
            un_normed_cond_ll_logprobs_new)
        log_weights_new = np.array(un_normed_cond_ll_logprobs_new) - np.array(un_normed_cond_ll_logprobs)
        log_weights_new = log_weights_new - np.max(log_weights_new)
        weights_new = np.exp(log_weights_new) / np.sum(np.exp(log_weights_new))

        # sample_efficiency = 1 / np.sum(weights_new ** 2)
        # if sample_efficiency < self.effective_n:
        #     random_samples = np.random.choice(next_state_samples.shape[0], size=self.n, replace=True, p=weights_new)
        #     next_state_samples = next_state_samples[random_samples]
        #     weights_new = np.ones(self.n) * (1 / self.n)
        # post_dist = GMMDist(weights_new, pred_state_marg.means, pred_state_marg.covs)
        # sampled_data = post_dist.sample(N)

        self.last_states = np.array(self.states)
        self.last_weights = np.array(self.weights)
        self.weights = weights_new
        self.states = next_state_samples
        # if self.collision_func(np.dot(weights_new, next_state_samples)):
        #     self.states = [self.collision_deviation(p) for p in next_state_samples]
        self.states = np.clip(self.states, self.limit[0], self.limit[1])
        self.state, self.cov = self.calculate_bs_moments()
