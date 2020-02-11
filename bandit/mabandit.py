import numpy as np
import random

# アーム定義
# ベルヌーイ分布のアーム
class BernoulliArm():
    def __init__(self, p):
        self.p = p

    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0
# 正規分布のアーム
class NormalArm():
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma

  def draw(self):
    return random.gauss(self.mu, self.sigma)

# 文脈を考慮したベルヌーイ分布のアーム
class ContextualBernoulliArm():
    def __init__(self, theta, noise = 0.01):
        self.theta = theta
        self.noise = noise

    def draw(self, x):
        f = np.dot(self.theta, x) + np.random.normal(loc=0, scale = self.noise)
        p = utils.sigmoid(f)
        if random.random() > p:
            return 0.0
        else:
            return 1.0

# 文脈を考慮した正規分布のアーム
class ContextualNormalArm():
    def __init__(self, theta, sigma, noise = 0.1):
        self.theta = theta
        self.sigma = sigma
        self.noise = noise

    def draw(selfs, x):
        mu = np.dot(self.theta, x) + np.random.normal(loc=0, scale = self.noise)
        return random.gauss(mu, self.sigma)

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

# Contextual bandit
# LinUCB
class LinUCB():
    def __init__(self, alpha, counts, values, probs, context_dim):
        self.alpha = alpha
        self.counts = counts
        self.values = values
        self.probs = probs
        self.context_dim = context_dim

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.probs = np.zeros(n_arms)

        self.A = [np.identity(self.context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(self.context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(self.context_dim) for _ in range(n_arms)]

    def select_arm(self, x):
        n_arms = len(self.counts)
        if min(self.counts) == 0:
            return np.argmin(self.counts)

        # アームのパラメータをアップデート
        self.theta = [np.dot(np.linalg.inv(self.A[arm]), self.b[arm]) for arm in range(n_arms)]
        bonus = [self.alpha * np.sqrt(float(np.dot(np.dot(x.T, np.linalg.inv(self.A[arm])), x)))
                    for arm in range(n_arms)]
        self.probs = [np.dot(self.theta[arm].T, x) + bonus[arm] for arm in range(n_arms)]
        return np.argmax(self.probs)

    def update(self, chosen_arm, reward, x):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

        self.A[chosen_arm] = self.A[chosen_arm] + np.dot(x, x.T)
        self.b[chosen_arm] = self.b[chosen_arm] + reward * x

# Lin Thompson Sampling
class LinTS():
    def __init__(self, counts, values, probs, context_dim, sigma_0=0.1, sigma=1):
        self.counts = counts
        self.values = values
        self.probs = probs
        self.context_dim = context_dim
        self.sigma_0 = sigma_0
        self.sigma = sigma

    def initialize(self, n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.probs = np.zeros(n_arms)

        self.A = [np.dot(self.sigma_0**2 / self.sigma**2, np.identity(self.context_dim)) for _ in range(n_arms)]
        self.b = [np.zeros(self.context_dim) for _ in range(n_arms)]
        self.theta = [np.zeros(self.context_dim) for _ in range(n_arms)]

    def select_arm(self, x):
        n_arms = len(self.counts)
        if min(self.counts) == 0:
            return np.argmin(self.counts)

        # アームのパラメータをアップデート
        self.theta = [np.random.multivariate_normal(np.dot(np.linalg.inv(self.A[arm]), self.b[arm]), self.sigma**2 * np.linalg.inv(self.A[arm])) for arm in range(n_arms)]
        self.probs = [np.dot(self.theta[arm].T, x) for arm in range(n_arms)]
        return np.argmax(self.probs)

    def update(self, chosen_arm, reward, x):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1 / float(n))*reward # これまでの報酬に今回の報酬を加重平均で加える
        self.values[chosen_arm] = new_value

        self.A[chosen_arm] = self.A[chosen_arm] + np.dot(x, x.T)
        self.b[chosen_arm] = self.b[chosen_arm] + reward * x

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

def test_contextual_mab_algorithm(algo, arms, X, num_sims, horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon)
    cumulative_regrets = np.zeros(num_sims * horizon)
    times = np.zeros(num_sims * horizon)
    is_best_arms = np.zeros(num_sims * horizon)
    for sim in range(num_sims):
        algo.initialize(len(arms))
        for t in range(horizon):
            idx = sim * horizon + t
            times[idx] = t + 1
            # アーム選択
            chosen_arm = algo.select_arm(X[t])
            chosen_arms[idx] = chosen_arm
            # 報酬獲得
            reward = arms[chosen_arm].draw(X[t])

            # 期待値最大だったか
            tmp_arm_mus = []
            for arm in arms:
                tmp_arm_mus.append(np.dot(arm.theta, X[t]))
            best_arm = np.argmax(tmp_arm_mus)
            is_best_arms[idx] = int(chosen_arm == best_arm)

            if t == 0:
                cumulative_rewards[idx] = reward
                cumulative_regrets[idx] = arms[best_arm].draw(X[t]) - reward
            else:
                cumulative_rewards[idx] = cumulative_rewards[idx - 1] + reward
                cumulative_regrets[idx] = cumulative_regrets[idx - 1] + arms[best_arm].draw(X[t]) - reward

            # 状態更新
            algo.update(chosen_arm, reward, X[t])
            # if sim == 1:
            #     print('%d: best_arm: %d, chosen_arm: %d' % (t, best_arm, chosen_arm))
        utils.progress_bar(sim, num_sims)
    return [times, chosen_arms, cumulative_rewards, is_best_arms, cumulative_regrets]
