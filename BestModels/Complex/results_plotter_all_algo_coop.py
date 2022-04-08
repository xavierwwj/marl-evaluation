import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

csv_paths = [
    "./DQN_PER_COOP/DQNTrainer_2022-04-02_08-22-32/DQNTrainer_foodenv_b7659_00000_0_2022-04-02_08-22-32/progress.csv",
    "./QMIX_COOP/QMIX/QMIX_grouped_foodenv_fabe5_00000_0_2022-04-01_23-20-23/progress.csv"
    ]
cases = ['Coop (Centralised)','Coop (Centralised)']
models = ['DQN_PER','QMIX']
optimal_val = 186

ax = None

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    # df['timesteps'] = df['iterations_since_restore'].values*100
    if ax == None:
        ax = df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i])
    else:
        df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i],ax=ax)

plt.hlines(optimal_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Cooperative \nEpisode Reward',color='r')
plt.xlabel("Timestep")
plt.ylabel("Mean Episode Reward (Centralized Env.)")
plt.legend(loc='lower right')
plt.show()