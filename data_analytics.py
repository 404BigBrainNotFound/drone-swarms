import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("swarm_monte_carlo_results.xlsx", sheet_name="Raw_Trial_Data")

algorithms = df['algorithm'].unique()

# PLOT 1
grouped = df.groupby(['algorithm', 'time_step'])['formation_error']
mean = grouped.mean()
std = grouped.std()

plt.figure()
for alg in algorithms:
    t = mean[alg].index
    m = mean[alg].values
    s = std[alg].values

    plt.plot(t, m, label=f"{alg} mean")

plt.xlabel("Time step")
plt.ylabel("Formation error")
plt.title("Formation Error Evolution")
plt.legend()
plt.show()

# PLOT 2
collisions = df.groupby(['algorithm', 'time_step'])['collision_count'].mean()

plt.figure()
for alg in algorithms:
    plt.plot(
        collisions[alg].index,
        collisions[alg].values,
        label=alg
    )

plt.xlabel("Time step")
plt.ylabel("Mean collision count")
plt.title("Collision Evolution Over Time")
plt.legend()
plt.show()

# PLOT 3
min_sep = df.groupby(['algorithm', 'time_step'])['min_separation'].mean()

plt.figure()
for alg in algorithms:
    plt.plot(
        min_sep[alg].index,
        min_sep[alg].values,
        label=alg
    )

plt.xlabel("Time step")
plt.ylabel("Mean minimum separation")
plt.title("Minimum Inter-Agent Separation Over Time")
plt.legend()
plt.show()

# PLOT 4
tradeoff = df.groupby(['algorithm', 'time_step']).mean(numeric_only=True)

plt.figure()
for alg in algorithms:
    plt.scatter(
        tradeoff.loc[alg, 'formation_error'],
        tradeoff.loc[alg, 'collision_count'],
        alpha=0.5,
        label=alg
    )

plt.xlabel("Formation error")
plt.ylabel("Collision count")
plt.title("Accuracy–Safety Tradeoff")
plt.legend()
plt.show()
