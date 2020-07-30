import numpy as np

"""
Adapted from: https://pylessons.com/CartPole-PER/
to include IS weights
"""

class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity

        self.tree = np.zeros(2 * capacity - 1)

        self.data_list = list(np.zeros(capacity))

    def add(self, priority, experience):
        tree_index = self.data_pointer + self.capacity - 1

        self.data_list[self.data_pointer] = experience

        self.update(tree_index, priority)

        self.data_pointer += 1

        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data_list[data_index]

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def used_up_capacity(self):
        return np.count_nonzero(self.tree[-self.capacity:])


class Memory:
    def __init__(self, capacity):
        self.PER_e = 0.01
        self.PER_a = 0.8
        self.PER_b = 0.4

        self.PER_b_increment_per_sampling = 0.001

        self.absolute_error_upper = 5.

        self.tree = SumTree(capacity)

    def store(self, experience):
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)

    def sample(self, n):
        minibatch = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        priority_segment = self.tree.total_priority / n

        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])

        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i)) ** b / max_weight = (N*P(i)) ** -b / max_weight
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            minibatch.append(experience)

        return b_idx, minibatch, b_ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors = [abs_error + self.PER_e for abs_error in abs_errors]
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
