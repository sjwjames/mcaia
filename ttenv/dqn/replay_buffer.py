import numpy as np
import random
import torch


# We'll implement the segment tree directly here to avoid dependencies
class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        
        https://en.wikipedia.org/wiki/Segment_tree
        
        Parameters
        ----------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            Mapping for combining elements (e.g. min, max, sum)
        neutral_element: obj
            Neutral element for the operation above.
            (e.g. float('inf') for min, 0 for sum)
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "Capacity must be a power of 2"
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying operation to the range [start, end)"""
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=lambda a, b: a + b,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns the sum in the range [start, end)"""
        return self.reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
            
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        
        Parameters
        ----------
        prefixsum: float
            upper bound on the sum of array prefix
            
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ..., arr[end])"""
        return self.reduce(start, end)


class ReplayBuffer:
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        device: torch.device
            PyTorch device to store tensors on
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.device = device

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        """Add a new experience to the buffer.
        
        Parameters
        ----------
        obs_t: numpy.ndarray
            Current observation
        action: int or float or numpy.ndarray
            Action taken
        reward: float
            Reward received
        obs_tp1: numpy.ndarray
            Next observation
        done: bool
            Whether the episode ended after this transition
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """Convert samples to pytorch tensors."""
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

        # Convert to PyTorch tensors hacking for now
        obses_t = torch.tensor(np.array(obses_t), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        obses_tp1 = torch.tensor(np.array(obses_tp1), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        return obses_t, actions, rewards, obses_tp1, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: torch.Tensor
            batch of observations
        act_batch: torch.Tensor
            batch of actions executed given obs_batch
        rew_batch: torch.Tensor
            rewards received as results of executing act_batch
        next_obs_batch: torch.Tensor
            next set of observations seen after executing act_batch
        done_mask: torch.Tensor
            done_mask[i] = 1.0 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, device):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        device: torch.device
            PyTorch device to store tensors on

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, device)
        assert alpha >= 0
        self._alpha = alpha

        # Find capacity as power of 2
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.add"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        Compared to ReplayBuffer.sample this also returns importance weights and idxes
        of sampled experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: torch.Tensor
            batch of observations
        act_batch: torch.Tensor
            batch of actions executed given obs_batch
        rew_batch: torch.Tensor
            rewards received as results of executing act_batch
        next_obs_batch: torch.Tensor
            next set of observations seen after executing act_batch
        done_mask: torch.Tensor
            done_mask[i] = 1.0 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: torch.Tensor
            Array of shape (batch_size,) and dtype torch.float32
            denoting importance weight of each sampled transition
        idxes: numpy.ndarray
            Array of shape (batch_size,) and dtype np.int32
            indices in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        # Convert to PyTorch tensors
        weights = torch.tensor(np.array(weights), dtype=torch.float32, device=self.device)
        encoded_sample = self._encode_sample(idxes)

        return tuple(list(encoded_sample) + [weights, np.array(idxes)])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        Sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class ParticleBeliefReplayBuffer(ReplayBuffer):
    def __init__(self, size, device):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        device: torch.device
            PyTorch device to store tensors on
        """
        super().__init__(size, device)
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.device = device

    def __len__(self):
        return len(self._storage)

    def add(self, target_belief_t, agent_info_t, action, reward, target_belief_t1, agent_info_t1, done):
        """Add a new experience to the buffer.

        Parameters
        ----------
        target_belief_t/1: numpy.ndarray
            particle beliefs
        agent_info_t/1: numpy.ndarray
            agent state and observed obstacle info
        action: int or float or numpy.ndarray
            Action taken
        reward: float
            Reward received
        done: bool
            Whether the episode ended after this transition
        """
        data = (target_belief_t, agent_info_t, action, reward, target_belief_t1, agent_info_t1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        """Convert samples to pytorch tensors."""
        actions, rewards, dones = [], [], []
        target_beliefs_t=torch.tensor([], dtype=torch.float32, device=self.device)
        agent_infos_t = torch.tensor([], dtype=torch.float32, device=self.device)
        target_beliefs_t1 = torch.tensor([], dtype=torch.float32, device=self.device)
        agent_infos_t1 = torch.tensor([], dtype=torch.float32, device=self.device)
        for i in idxes:
            data = self._storage[i]
            target_belief_t, agent_info_t, action, reward, target_belief_t1, agent_info_t1, done = data
            target_beliefs_t=torch.cat((target_beliefs_t,target_belief_t),dim=0)
            agent_infos_t=torch.cat((agent_infos_t,agent_info_t),dim=0)
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            target_beliefs_t1=torch.cat((target_beliefs_t1, target_belief_t1),dim=0)
            agent_infos_t1=torch.cat((agent_infos_t1, agent_info_t1),dim=0)
            dones.append(done)

        # Convert to PyTorch tensors hacking for now
        actions = torch.tensor(np.array(actions), dtype=torch.int64, device=self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32, device=self.device)

        return target_beliefs_t, agent_infos_t, actions, rewards, target_beliefs_t1, agent_infos_t1, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: torch.Tensor
            batch of observations
        act_batch: torch.Tensor
            batch of actions executed given obs_batch
        rew_batch: torch.Tensor
            rewards received as results of executing act_batch
        next_obs_batch: torch.Tensor
            next set of observations seen after executing act_batch
        done_mask: torch.Tensor
            done_mask[i] = 1.0 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
