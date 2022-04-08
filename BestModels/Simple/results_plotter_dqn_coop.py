import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

csv_paths = [
    "./DQN_COOP/DQNTrainer_2022-04-01_04-12-40/DQNTrainer_foodenv_a568c_00000_0_2022-04-01_04-12-41/progress.csv",
    "./DQN_PER_COOP/DQNTrainer_2022-04-01_04-11-22/DQNTrainer_foodenv_76743_00000_0_2022-04-01_04-11-22/progress.csv"
    ]
cases = ['Centralized', 'Centralized']
models = ['DQN', 'DQN (PER)']
optimal_val = 34

ax = None

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    # df['timesteps'] = df['iterations_since_restore'].values*100
    if ax == None:
        ax = df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i])
    else:
        df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i],ax=ax)

plt.hlines(optimal_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Cooperative Episode Reward',color='r')
plt.xlabel("Timestep")
plt.ylabel("Mean Episode Reward (Centralised Env.)")
plt.legend(loc='center right')
plt.show()