import numpy as np
import pandas as pd

# --- Import Behavior Classes (Assuming these exist in your directory) ---
from behaviors.flocking_behavior_algorithm import FlockingBehavior
from behaviors.consensus_algorithm import ConsensusAlgorithm
from behaviors.collision_avoidance_algorithm import CollisionAvoidanceAlgorithm
from behaviors.formation_control_algorithm import FormationControlAlgorithm

# --- Simulation Parameters ---
N_DRONES = 20
TOTAL_STEPS = 500  # Reduced for speed during MC trials; increase as needed
N_TRIALS = 30      # Number of Monte Carlo iterations
COLLISION_THRESHOLD = 1.0
FORMATION_TYPE = 'square'
EPSILON = 0.1 

# --- Updated Drone Class (Inherited from your original) ---
class ExperimentDrone:
    def __init__(self, position, index):
        self.position = np.array(position, dtype=float)
        self.index = index
        self.target_position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)

    def update_position(self, neighbors, behavior_algorithms):
        position_updates = []
        for algorithm in behavior_algorithms:
            if type(algorithm).__name__ == "FlockingBehavior":
                self.velocity = algorithm.apply(self, neighbors)
                position_updates.append(self.position.copy() + self.velocity) 
            else:
                neighbor_positions = [d.get_position() for d in neighbors]
                proposed_pos = algorithm.apply(self, neighbor_positions, self.position.copy())
                position_updates.append(proposed_pos)

        if position_updates:
            self.position = np.mean(position_updates, axis=0)

        for alg in behavior_algorithms:
            if isinstance(alg, FormationControlAlgorithm):
                neighbor_positions = [d.get_position() for d in neighbors]
                self.target_position = alg.apply(self, neighbor_positions, self.position.copy())

    def get_position(self):
        return self.position

# --- Helper Functions ---

def initialize_swarm(num_drones, seed):
    """Initializes the swarm with a specific seed for reproducibility within trials."""
    np.random.seed(seed)
    drones = []
    for i in range(num_drones):
        pos = np.random.rand(3) * 20 # Increased space to see convergence better
        drones.append(ExperimentDrone(pos, i))
    return drones

def calculate_metrics(drones, collision_threshold):
    positions = np.array([d.position for d in drones])
    targets = np.array([d.target_position for d in drones])
    
    errors = np.linalg.norm(positions - targets, axis=1)
    formation_error = np.mean(errors)

    min_separation = float('inf')
    collision_count = 0
    num_drones = len(drones)

    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_separation: min_separation = dist
            if dist < collision_threshold: collision_count += 1
    
    return {
        "formation_error": formation_error,
        "min_separation": min_separation if min_separation != float('inf') else 0.0,
        "collision_count": collision_count
    }

def run_experiment(algorithm_name, drones, total_steps, trial_id):
    if algorithm_name == 'Consensus':
        behaviors = [ConsensusAlgorithm(EPSILON), CollisionAvoidanceAlgorithm(COLLISION_THRESHOLD), FormationControlAlgorithm(FORMATION_TYPE)]
    elif algorithm_name == 'Flocking':
        behaviors = [FlockingBehavior(), CollisionAvoidanceAlgorithm(COLLISION_THRESHOLD), FormationControlAlgorithm(FORMATION_TYPE)]
    
    trial_logs = []
    for t in range(total_steps):
        for drone in drones:
            neighbors = [d for d in drones if d != drone]
            drone.update_position(neighbors, behaviors)

        metrics = calculate_metrics(drones, COLLISION_THRESHOLD)
        metrics.update({"time_step": t, "algorithm": algorithm_name, "trial": trial_id})
        trial_logs.append(metrics)

    return trial_logs

# --- Monte Carlo Execution ---

if __name__ == "__main__":
    all_results = []
    
    print(f"Starting Monte Carlo Simulation with {N_TRIALS} trials...")

    for trial in range(N_TRIALS):
        # Generate a unique seed for this trial
        trial_seed = np.random.randint(0, 10000)
        print(f"--- Trial {trial + 1}/{N_TRIALS} (Seed: {trial_seed}) ---")

        # 1. Run Consensus
        drones_c = initialize_swarm(N_DRONES, trial_seed)
        res_c = run_experiment('Consensus', drones_c, TOTAL_STEPS, trial)
        all_results.extend(res_c)

        # 2. Run Flocking
        drones_f = initialize_swarm(N_DRONES, trial_seed)
        res_f = run_experiment('Flocking', drones_f, TOTAL_STEPS, trial)
        all_results.extend(res_f)

    # Convert to DataFrame
    final_df = pd.DataFrame(all_results)

    # Create a Summary (Average metrics over all trials per time step)
    summary_df = final_df.groupby(['algorithm', 'time_step']).agg({
        'formation_error': ['mean', 'std'],
        'collision_count': ['mean', 'sum'],
        'min_separation': 'mean'
    }).reset_index()

    # --- Save to Excel with Multiple Sheets ---
    output_filename = "swarm_monte_carlo_results.xlsx"
    with pd.ExcelWriter(output_filename) as writer:
        summary_df.to_excel(writer, sheet_name='Summary_Statistics')
        # We only save a sample or the full raw data if it's not too large
        # Excel has a limit of ~1 million rows.
        final_df.to_excel(writer, sheet_name='Raw_Trial_Data', index=False)

    print(f"\nSimulation complete. Data saved to {output_filename}")
    