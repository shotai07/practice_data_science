import numpy as np
import random

# アーム定義
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

# アルゴリズム実装
# ベースライン
class RandomSelect():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        return random.randint(0, len(self.values) - 1)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

# εグリーディ法
class EpsilonGreedy():
    def __init__(self, epsilon, counts, values):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        if random.random() > self.epsilon:
            return np.argmax(self.values)
        else:
            return random.randint(0, len(self.values) - 1)


    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

# UCB法
class UCB():
    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        n_arms = len(self.counts)
        if min(self.counts) == 0:
            return np.argmin(self.counts)

        total_counts = sum(self.counts)
        bonus = np.sqrt((np.log(np.array(total_counts))) / 2 / np.array(self.counts))
        ucb_values = np.array(self.values) + bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

# Thompson Sampling
class ThompsonSampling():
    def __init__(self, counts_alpha, counts_beta, values):
        self.counts_alpha = counts_alpha
        self.counts_beta = counts_beta
        self.alpha = 1
        self.beta = 1
        self.values = values

    def initialize(self, n_arms):
        self.counts_alpha = np.zeros(n_arms)
        self.counts_beta = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):
        theta = [(arm, random.betavariate(self.counts_alpha[arm] + self.alpha, self.counts_beta[arm] + self.beta)) for arm in range(len(self.counts_alpha))]
        theta = sorted(theta, key=lambda x: x[1])
        return theta[-1][0]

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.counts_alpha[chosen_arm] += 1
        else:
            self.counts_beta[chosen_arm] += 1

        n = self.counts_alpha[chosen_arm] + self.counts_beta[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

# テスト用関数

import utils

def test_mab_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            idx = sim * horizon + t
            times[idx] = t + 1
            # アーム選択
            chosen_arm = algo.select_arm()
            chosen_arms[idx] = chosen_arm
            # 報酬獲得
            reward = arms[chosen_arm].draw()
            if t == 0:
                cumulative_rewards[idx] = reward
            else:
                cumulative_rewards[idx] = cumulative_rewards[idx - 1] + reward

            # 状態更新
            algo.update(chosen_arm, reward)
        utils.progress_bar(sim, num_sims)
    return [times, chosen_arms, cumulative_rewards]
