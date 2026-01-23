"""Deep ADFQ learning graph - PyTorch Implementation

This code was modified from the original TensorFlow implementation to use PyTorch.
The major difference is the output size of the network (2*num_actions) and the action selection methods.
"""

import torch
import torch.nn as nn
import numpy as np


def build_act(model, num_actions, eps=0.0, stochastic=True):
    """Creates the act function for action selection.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to use for action selection
    num_actions : int
        Number of possible actions
    eps : float
        Exploration rate
    stochastic : bool
        Whether to use stochastic action selection

    Returns
    -------
    act : function
        Function to select an action given an observation
    """

    def act(obs, stochastic=stochastic, update_eps=-1):
        if update_eps >= 0:
            nonlocal eps
            eps = update_eps

        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            q_values = model(obs)  # Shape: (batch_size, num_actions*2)
            q_means = q_values[:, :num_actions]
            deterministic_actions = torch.argmax(q_means, dim=1)

            if stochastic and eps > 0:
                batch_size = obs.shape[0]
                random_actions = torch.randint(0, num_actions, (batch_size,))
                chose_random = torch.rand(batch_size) < eps
                output_actions = torch.where(chose_random, random_actions, deterministic_actions)
            else:
                output_actions = deterministic_actions

        return output_actions.numpy()

    return act


def build_act_greedy(model, num_actions, eps=0.0):
    """Creates the act function for a simple fixed epsilon greedy policy.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to use for action selection
    num_actions : int
        Number of possible actions
    eps : float
        Fixed exploration rate

    Returns
    -------
    act : function
        Function to select an action given an observation
    """

    def act(obs, stochastic=True):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            q_values = model(obs)  # Shape: (batch_size, num_actions*2)
            q_means = q_values[:, :num_actions]
            deterministic_actions = torch.argmax(q_means, dim=1)

            if stochastic and eps > 0:
                batch_size = obs.shape[0]
                random_actions = torch.randint(0, num_actions, (batch_size,))
                chose_random = torch.rand(batch_size) < eps
                output_actions = torch.where(chose_random, random_actions, deterministic_actions)
            else:
                output_actions = deterministic_actions

        return output_actions.numpy()

    return act


def build_act_bayesian(model, num_actions):
    """Creates the act function for Bayesian sampling.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to use for action selection
    num_actions : int
        Number of possible actions

    Returns
    -------
    act : function
        Function to select an action given an observation
    """

    def act(obs, stochastic=True, update_eps=-1):
        with torch.no_grad():
            obs = torch.FloatTensor(obs)
            q_values = model(obs)  # Shape: (batch_size, num_actions*2)
            q_means = q_values[:, :num_actions]
            q_sds = torch.exp(-q_values[:, num_actions:])

            # Sample from normal distribution
            samples = torch.randn_like(q_means) * q_sds + q_means
            output_actions = torch.argmax(samples, dim=1)

        return output_actions.numpy()

    return act


def build_train(model, target_model, optimizer, num_actions, gamma=0.99,
                grad_norm_clipping=None, sdMin=1e-5, sdMax=1e5,
                test_eps=0.05, act_policy='egreedy', lr_init=0.001,
                lr_decay_factor=0.99, lr_growth_factor=1.001, tau=0.001):
    """Creates the training function.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    target_model : nn.Module
        The target network for Q-learning
    optimizer : torch.optim.Optimizer
        The optimizer to use
    num_actions : int
        Number of possible actions
    gamma : float
        Discount factor
    grad_norm_clipping : float or None
        Clip gradient norms to this value
    sdMin, sdMax : float
        Min/max standard deviation values
    test_eps : float
        Exploration rate for testing
    act_policy : str
        Action selection policy ('egreedy' or 'bayesian')
    lr_init : float
        Initial learning rate
    lr_decay_factor : float
        Learning rate decay factor
    lr_growth_factor : float
        Learning rate growth factor
    tau : float
        Target network update rate

    Returns
    -------
    train : function
        Function to train the model on a batch of experiences
    """

    def train(obs_t, action, reward, obs_tp1, done, weight=None):
        # Convert inputs to tensors
        obs_t = torch.FloatTensor(obs_t)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        obs_tp1 = torch.FloatTensor(obs_tp1)
        done = torch.FloatTensor(done)
        if weight is not None:
            weight = torch.FloatTensor(weight)

        # Get current Q values
        q_values = model(obs_t)  # Shape: (batch_size, num_actions*2)
        q_means = q_values[:, :num_actions]
        q_sds = torch.exp(-q_values[:, num_actions:])

        # Get Q values for taken actions
        q_taken = q_means.gather(1, action.unsqueeze(1)).squeeze(1)
        sd_taken = q_sds.gather(1, action.unsqueeze(1)).squeeze(1)

        # Get target Q values
        with torch.no_grad():
            target_q_values = target_model(obs_tp1)  # Shape: (batch_size, num_actions*2)
            target_q_means = target_q_values[:, :num_actions]
            target_q_sds = torch.exp(-target_q_values[:, num_actions:])

            # For Bayesian policy, sample from target distribution
            if act_policy == 'bayesian':
                target_samples = torch.randn_like(target_q_means) * target_q_sds + target_q_means
                target_q = torch.max(target_samples, dim=1)[0]
            else:
                target_q = torch.max(target_q_means, dim=1)[0]

            target_q = reward + (1 - done) * gamma * target_q

        # Compute loss
        td_error = q_taken - target_q
        loss = torch.mean(weight * td_error.pow(2) if weight is not None else td_error.pow(2))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        if grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clipping)
        optimizer.step()

        return td_error.detach().numpy()

    return train


def update_target(target_model, model, tau=0.001):
    """Update target network parameters.

    Parameters
    ----------
    target_model : nn.Module
        The target network to update
    model : nn.Module
        The source network
    tau : float
        Update rate
    """
    for target_param, param in zip(target_model.parameters(), model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)