import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

csv_paths = [
    "./QMIX_COMP/QMIX/QMIX_grouped_foodenv_1ae99_00000_0_2022-04-02_03-24-40/progress.csv"
    ]
cases = ['Coop (De-centralised)']
models = ['QMIX']
# optimal_val = 8

ax = None

for i in range(len(csv_paths)):
    df = pd.read_csv(csv_paths[i])
    # df['timesteps'] = df['iterations_since_restore'].values*100
    if ax == None:
        ax = df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i])
    else:
        df.plot(x='timesteps_total', y='episode_reward_mean',label=models[i],ax=ax)

# plt.hlines(optimal_val,0,df['timesteps_total'].values[-1],linestyles='dashed',label='Optimal Episode Reward',color='r')
plt.xlabel("Timestep")
plt.ylabel("Mean Episode Reward (De-centralized Env.)")
plt.legend(loc='center right')
plt.show()