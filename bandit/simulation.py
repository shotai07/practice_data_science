import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mabandit as mab
import random

# 諸条件設定
n_arms = 10 # アーム数

arm_mus = [] # アームの期待値
random.seed(0)
for i in range(n_arms):
    arm_mus.append(random.random())

print('# Each Arm mu:')
for i in range(n_arms):
    print('%d: %.2f' % (i, arm_mus[i]))
arms = pd.Series(map(lambda x: mab.BernoulliArm(x), arm_mus))
max_arm = np.argmax(arm_mus)

epsilon = 0.2 # εグリーディのパラメータ
sim_num = 500 # シミュレーション回数
time = 10000 # アーム選択の試行回数

# アルゴリズム設定
algos = []
algos.append(mab.RandomSelect([], []))
algos.append(mab.EpsilonGreedy(epsilon, [], []))
algos.append(mab.UCB([], []))
algos.append(mab.ThompsonSampling([], [], []))

# 描画設定
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
heights = []

# シミュレーション実行
for a in algos:
    print('\n')
    print(a.__class__.__name__)
    print('----------------------------------------------------------------------')
    a.initialize(n_arms)
    result = mab.test_mab_algorithm(a, arms, sim_num, time)

    df_result = pd.DataFrame({'times': result[0], 'chosen_arms': result[1], 'cumulative_rewards': result[2]})
    df_result['is_best_arm'] = (df_result['chosen_arms'] == max_arm).astype(int)
    grouped = df_result['is_best_arm'].groupby(df_result['times'])
    ax1.plot(grouped.mean(), label=a.__class__.__name__)
    heights.append(df_result['cumulative_rewards'].iloc[-1])

# 描画
ax1.set_title('Compare model - best arm rate')
ax1.set_xlabel('Time')
ax1.set_ylabel('Best arm rate')
ax1.legend(loc='best')

plt_label = ['Random', 'Epsilon\nGreedy', 'UCB', 'Thompson\nSampling']
plt_colot = ['deep', 'muted', 'pastel', 'bright']
ax2.bar(range(1,len(plt_label) + 1), heights, color=sns.color_palette()[:len(plt_label)], align='center')
ax2.set_xticks(range(1,len(plt_label) + 1))
ax2.set_xticklabels(plt_label)
ax2.set_label('random_select')
ax2.set_ylabel('cumulative_rewards')
ax2.set_title('Compare model - cumulative rewards')
plt.show()
