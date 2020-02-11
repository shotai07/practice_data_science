import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mabandit as mab
import random

# 諸条件設定
n_arms = 10 # アーム数

arm_thetas = [] # アームの期待値
random.seed(0)
for i in range(n_arms):
    arm_thetas.append(np.array([random.uniform(0,0.5), random.uniform(0.5,1.0)]))

print('# Each Arm theta:')
for i in range(n_arms):
    tmp_theta = [str(v) for v in arm_thetas[i]]
    print('%d: %s' % (i, ', '.join(tmp_theta)))
arms = pd.Series(map(lambda x: mab.ContextualBernoulliArm(x), arm_thetas))

epsilon = 0.2 # εグリーディのパラメータ
sim_num = 100 # シミュレーション回数
time = 1000 # アーム選択の試行回数

X = []
x_dict = {0: np.array([1,0]), 1: np.array([0,1])}
for i in range(time):
    x = random.randint(0,1)
    X.append(x_dict[x])

# アルゴリズム設定
algos = []
algos.append(mab.LinUCB(1.0, [], [], [], len(X[0])))
algos.append(mab.LinTS([], [], [], len(X[0])))

# 描画設定
fig = plt.figure()
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
heights = []

# シミュレーション実行
for a in algos:
    print('\n')
    print(a.__class__.__name__)
    print('----------------------------------------------------------------------')
    a.initialize(n_arms)
    result = mab.test_contextual_mab_algorithm(a, arms, X, sim_num, time)

    df_result = pd.DataFrame({'times': result[0], 'chosen_arms': result[1], 'cumulative_rewards': result[2], 'is_best_arm': result[3], 'cumulative_regrets': result[4]})
    grouped = df_result['is_best_arm'].groupby(df_result['times'])
    ax1.plot(grouped.mean(), label=a.__class__.__name__)
    heights.append(df_result['cumulative_rewards'].iloc[-1])
    grouped2 = df_result['cumulative_regrets'].groupby(df_result['times'])
    ax3.plot(grouped2.mean(), label = a.__class__.__name__)

# 描画
ax1.set_title('Compare model - best arm rate')
ax1.set_xlabel('Time')
ax1.set_ylabel('Best arm rate')
ax1.legend(loc='best')

plt_label = ['Contextual LinUCB', 'Contextual LinTS']
plt_colot = ['deep', 'muted']
ax2.bar(range(1,len(plt_label) + 1), heights, color=sns.color_palette()[:len(plt_label)], align='center')
ax2.set_xticks(range(1,len(plt_label) + 1))
ax2.set_xticklabels(plt_label)
ax2.set_label('random_select')
ax2.set_ylabel('cumulative_rewards')
ax2.set_title('Compare model - cumulative rewards')

ax3.set_title('Compare model - regrets')
ax3.set_xlabel('Time')
ax3.set_ylabel('Regrets')
ax3.legend(loc='best')

plt.show()
