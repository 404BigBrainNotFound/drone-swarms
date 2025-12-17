# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# def plot_results(csv_file='swarm_comparison_data.csv'):
#     # Load data
#     df = pd.read_csv(csv_file)

#     # Set style
#     sns.set(style="whitegrid")
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     # 1. Formation Error Line Plot
#     sns.lineplot(data=df, x='time_step', y='formation_error', hue='algorithm', ax=axes[0], linewidth=2.5)
#     axes[0].set_title('Swarm Formation Accuracy Over Time', fontsize=14)
#     axes[0].set_ylabel('Formation Error (Lower is Better)')
#     axes[0].set_xlabel('Simulation Step')

#     # 2. Collision Count Bar Plot
#     # We take the mean collision count per algorithm
#     collision_summary = df.groupby('algorithm')['collision_count'].mean().reset_index()
#     sns.barplot(data=collision_summary, x='algorithm', y='collision_count', ax=axes[1], palette=['blue', 'orange'])
#     axes[1].set_title('Average Collisions per Step (Safety)', fontsize=14)
#     axes[1].set_ylabel('Avg. Collision Count (Lower is Better)')

#     plt.tight_layout()
#     plt.savefig('swarm_results_plot.png')
#     plt.show()

# if __name__ == "__main__":
#     # Ensure you have seaborn installed: pip install seaborn
#     plot_results()


# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# def plot_monte_carlo_results(excel_file='swarm_monte_carlo_results.xlsx'):
#     df = pd.read_excel(excel_file, sheet_name='Raw_Trial_Data')
#     sns.set_theme(style="ticks")
    
#     # Create 3 subplots now
#     fig, axes = plt.subplots(1, 3, figsize=(22, 6))

#     # 1. Formation Error (Accuracy)
#     sns.lineplot(data=df, x='time_step', y='formation_error', hue='algorithm', 
#                  ax=axes[0], errorbar=('ci', 95))
#     axes[0].set_title('A. Formation Convergence', fontsize=14, fontweight='bold')
#     axes[0].set_ylabel('Error (Lower is Better)')

#     # 2. Collision Profile (Safety)
#     sns.lineplot(data=df, x='time_step', y='collision_count', hue='algorithm', 
#                  ax=axes[1], errorbar=('ci', 95), palette='Set2')
#     axes[1].set_title('B. Safety Profile (Collisions)', fontsize=14, fontweight='bold')
    
#     # 3. Settling Time Analysis (Stability)
#     # We plot the derivative (rate of change) of the error to see when it stops changing
#     df['error_change'] = df.groupby(['algorithm', 'trial'])['formation_error'].diff().abs()
#     sns.lineplot(data=df, x='time_step', y='error_change', hue='algorithm', 
#                  ax=axes[2], palette='flare')
#     axes[2].set_title('C. Stability (Rate of Change)', fontsize=14, fontweight='bold')
#     axes[2].set_ylabel('Delta Error')
#     axes[2].set_ylim(0, 0.5) # Zoom in on the tail end

#     # Aesthetic clean up
#     for ax in axes:
#         ax.grid(True, linestyle='--', alpha=0.4)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#     plt.tight_layout()
#     plt.savefig('swarm_comprehensive_analysis.pdf', format='pdf')
#     plt.show()

#     # Calculate Numeric Settling Time
#     # Defined as the first time step where error stays below a threshold
#     threshold = df['formation_error'].min() * 1.10 
#     for alg in df['algorithm'].unique():
#         alg_data = df[df['algorithm'] == alg].groupby('time_step')['formation_error'].mean()
#         settled_steps = alg_data[alg_data < threshold]
#         if not settled_steps.empty:
#             print(f"{alg} Settling Time: Step {settled_steps.index[0]}")

# if __name__ == "__main__":
#     plot_monte_carlo_results()



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
    plt.fill_between(t, m - s, m + s, alpha=0.2)

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
