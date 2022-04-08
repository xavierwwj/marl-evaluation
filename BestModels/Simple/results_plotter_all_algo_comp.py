import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

csv_paths = [
    "./DQN_PER_COMP/DQNTrainer_2022-04-01_04-14-02/DQNTrainer_foodenv_d62fa_00000_0_2022-04-01_04-14-02/progress.csv",
    "./QMIX_COMP/QMIX/QMIX_grouped_foodenv_2c94f_00000_0_2022-04-01_05-13-43/progress.csv",
    "./MADDPG_COMP/contrib/MADDPG/contrib_MADDPG_foodenv_30cc8_00000_0_2022-04-01_05-28-09/progress.csv"
    ]
cases = ['Comp', 'Comp', 'Comp']
models = ['DQN (PER)', 'QMIX', 'MADDPG']
optimal_greedy_val = 8
optimal_coop_val = 14

ax = None

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    # df['timesteps'] = df['iterations_since_restore'].values*100
    if ax == None:
        ax = df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i])
    else:
        df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i],ax=ax)

plt.hlines(optimal_greedy_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Competitive \nEpisode Reward',color='r')
plt.hlines(optimal_coop_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Cooperative \nEpisode Reward',color='y')
plt.xlabel("Timestep")
plt.ylabel("Mean Episode Reward (De-centralised Env.)")
plt.legend(loc='upper left')
plt.show()