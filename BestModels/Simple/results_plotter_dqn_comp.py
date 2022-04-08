import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

csv_paths = [
    "./DQN_COMP/DQNTrainer_2022-04-01_04-14-52/DQNTrainer_foodenv_f3a64_00000_0_2022-04-01_04-14-52/progress.csv",
    "./DQN_PER_COMP/DQNTrainer_2022-04-01_04-14-02/DQNTrainer_foodenv_d62fa_00000_0_2022-04-01_04-14-02/progress.csv"
    ]
cases = ['De-centralized', 'De-centralized']
models = ['DQN', 'DQN (PER)']
optimal_val = 8

ax = None

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    # df['timesteps'] = df['iterations_since_restore'].values*100
    if ax == None:
        ax = df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i])
    else:
        df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i],ax=ax)

plt.hlines(optimal_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Competitive Episode Reward',color='r')
plt.xlabel("Timestep")
plt.ylabel("Mean Episode Reward (De-centralized Env.)")
plt.legend(loc='center right')
plt.show()