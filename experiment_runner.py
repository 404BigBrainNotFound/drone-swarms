import numpy as np
import pandas as pd
import time

# --- Import Behavior Classes ---
from behaviors.flocking_behavior_algorithm import FlockingBehavior
from behaviors.consensus_algorithm import ConsensusAlgorithm
from behaviors.collision_avoidance_algorithm import CollisionAvoidanceAlgorithm
from behaviors.formation_control_algorithm import FormationControlAlgorithm


# --- Simulation Parameters ---
N_DRONES = 20
TOTAL_STEPS = 1000
COLLISION_THRESHOLD = 1.0
FORMATION_TYPE = 'square'
EPSILON = 0.1  # Consensus rate

# --- Updated Drone Class ---
class ExperimentDrone:
    """
    A specialized Drone class for the experiment that supports both 
    velocity-based (Flocking) and position-based (Consensus) updates.
    """
    def __init__(self, position, index):
        self.position = np.array(position, dtype=float)
        self.index = index
        self.target_position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)  # Essential for flocking

    def update_position(self, neighbors, behavior_algorithms):
        """
        Updates position by averaging position-based algorithms and adding 
        velocity-based updates if present.
        """
        position_updates = []
        
        # 1. Apply Flocking (Velocity-based)
        for algorithm in behavior_algorithms:
            if type(algorithm).__name__ == "FlockingBehavior":
                # Flocking returns a velocity vector
                self.velocity = algorithm.apply(self, neighbors)
                # Propose new position based on velocity
                position_updates.append(self.position.copy() + self.velocity) 
            else:
                # 2. Apply Position-based algorithms (Consensus, Collision, Formation)
                # Extract neighbor positions for these algorithms
                neighbor_positions = [d.get_position() for d in neighbors]
                proposed_pos = algorithm.apply(self, neighbor_positions, self.position.copy())
                position_updates.append(proposed_pos)

        # 3. Calculate average of all proposed positions
        if position_updates:
            new_position = np.mean(position_updates, axis=0)
            self.position = new_position

        # 4. Update target position for metric tracking
        # We track the target based on the FormationControlAlgorithm
        for alg in behavior_algorithms:
            if isinstance(alg, FormationControlAlgorithm):
                neighbor_positions = [d.get_position() for d in neighbors]
                self.target_position = alg.apply(self, neighbor_positions, self.position.copy())

    def get_position(self):
        return self.position

# --- Helper Functions ---

def initialize_swarm(num_drones):
    """
    Initializes the swarm with repeatable random positions.
    """
    np.random.seed(42)  # Ensure identical start for both experiments
    drones = []
    for i in range(num_drones):
        # Random start positions within a 10x10x10 cube
        pos = np.random.rand(3) * 10
        drones.append(ExperimentDrone(pos, i))
    return drones

def calculate_metrics(drones, collision_threshold):
    """
    Calculates performance metrics for the current time step.
    
    Metrics:
    1. Formation Error (Mean distance to target)
    2. Min Separation Distance
    3. Collision Count
    """
    positions = np.array([d.position for d in drones])
    targets = np.array([d.target_position for d in drones])
    
    # 1. Formation Error
    # Average Euclidean distance between current pos and target pos
    errors = np.linalg.norm(positions - targets, axis=1)
    formation_error = np.mean(errors)

    # Calculate pairwise distances for separation and collisions
    min_separation = float('inf')
    collision_count = 0
    num_drones = len(drones)

    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            dist = np.linalg.norm(positions[i] - positions[j])
            
            if dist < min_separation:
                min_separation = dist
            
            if dist < collision_threshold:
                collision_count += 1
    
    if min_separation == float('inf'):
        min_separation = 0.0

    return {
        "formation_error": formation_error,
        "min_separation": min_separation,
        "collision_count": collision_count
    }

def run_experiment(algorithm_name, drones, total_steps):
    """
    Runs the simulation loop for a specific algorithm configuration.
    """
    print(f"Starting experiment: {algorithm_name}...")
    
    # Configure behavior list based on the algorithm name
    if algorithm_name == 'Consensus':
        behaviors = [
            ConsensusAlgorithm(EPSILON),
            CollisionAvoidanceAlgorithm(COLLISION_THRESHOLD),
            FormationControlAlgorithm(FORMATION_TYPE)
        ]
    elif algorithm_name == 'Flocking':
        behaviors = [
            FlockingBehavior(),
            CollisionAvoidanceAlgorithm(COLLISION_THRESHOLD),
            FormationControlAlgorithm(FORMATION_TYPE)
        ]
    else:
        raise ValueError("Unknown algorithm name")

    logs = []

    for t in range(total_steps):
        # 1. Update positions
        # Create a snapshot of current neighbors for the update step
        for drone in drones:
            neighbors = [d for d in drones if d != drone]
            drone.update_position(neighbors, behaviors)

        # 2. Calculate metrics
        metrics = calculate_metrics(drones, COLLISION_THRESHOLD)
        
        # 3. Log data
        log_entry = {
            "time_step": t,
            "algorithm": algorithm_name,
            **metrics
        }
        logs.append(log_entry)
        
        # Optional: Print progress every 100 steps
        if t % 100 == 0:
            print(f"  [{algorithm_name}] Step {t}/{total_steps} | Error: {metrics['formation_error']:.2f}")

    print(f"Finished {algorithm_name}.\n")
    return pd.DataFrame(logs)

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Run Consensus-Based Experiment
    drones_consensus = initialize_swarm(N_DRONES)
    df_consensus = run_experiment('Consensus', drones_consensus, TOTAL_STEPS)

    # 2. Run Bio-Inspired Flocking Experiment
    # Re-initialize to ensure exact same starting positions
    drones_flocking = initialize_swarm(N_DRONES)
    df_flocking = run_experiment('Flocking', drones_flocking, TOTAL_STEPS)

    # 3. Combine and Save Results
    final_df = pd.concat([df_consensus, df_flocking], ignore_index=True)
    
    output_filename = "swarm_comparison_data.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"Simulation complete. Data saved to {output_filename}")
    print("Summary:")
    print(final_df.groupby('algorithm')[['formation_error', 'collision_count']].mean())