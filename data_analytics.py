import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(csv_file='swarm_comparison_data.csv'):
    # Load data
    df = pd.read_csv(csv_file)

    # Set style
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Formation Error Line Plot
    sns.lineplot(data=df, x='time_step', y='formation_error', hue='algorithm', ax=axes[0], linewidth=2.5)
    axes[0].set_title('Swarm Formation Accuracy Over Time', fontsize=14)
    axes[0].set_ylabel('Formation Error (Lower is Better)')
    axes[0].set_xlabel('Simulation Step')

    # 2. Collision Count Bar Plot
    # We take the mean collision count per algorithm
    collision_summary = df.groupby('algorithm')['collision_count'].mean().reset_index()
    sns.barplot(data=collision_summary, x='algorithm', y='collision_count', ax=axes[1], palette=['blue', 'orange'])
    axes[1].set_title('Average Collisions per Step (Safety)', fontsize=14)
    axes[1].set_ylabel('Avg. Collision Count (Lower is Better)')

    plt.tight_layout()
    plt.savefig('swarm_results_plot.png')
    plt.show()

if __name__ == "__main__":
    # Ensure you have seaborn installed: pip install seaborn
    plot_results()