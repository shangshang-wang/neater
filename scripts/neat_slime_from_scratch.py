# Import necessary libraries
import neat
import numpy as np
import os
import pickle
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Adjust path to NEAT
import utils.visualize as visualize # Assuming visualize.py contains necessary plotting/drawing functions
import slimevolleygym
import gym
import gym.wrappers
import multiprocessing

import logging

# --- Configuration ---
CONFIG_PATH = os.path.join('task', 'config_slime') # Path to the NEAT config file
SAVE_INTERVAL = 50 # Save the best genome every 50 generations
OUTPUT_DIR = f'results/run_0'
BEST_GENOME_PREFIX = 'best_genome_gen_'
FINAL_WINNER_FILENAME = 'final_winner.pickle'
CHECKPOINT_PREFIX = 'neat-checkpoint-'

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# # --- Environment Setup ---
# # Suppress warnings and logging for cleaner output
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)
# import warnings
# from pkg_resources import PkgResourcesDeprecationWarning
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)
# warnings.filterwarnings("ignore", category=FutureWarning) # Ignore potential future gym warnings

# Create the SlimeVolley environment
try:
    # Try newer Gym API registration if available
    env = gym.make('SlimeVolley-v0', render_mode=None)
except TypeError:
    # Fallback for older gym/slimevolleygym versions
    env = gym.make('SlimeVolley-v0')
print(f"Environment Observation Space: {env.observation_space}")
print(f"Environment Action Space: {env.action_space}")


# --- NEAT Setup ---

# Custom Genome class (if needed, otherwise use neat.DefaultGenome)
class SlimeGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)

    def __str__(self):
        # Optional: Custom string representation
        return f"SlimeGenome {self.key}, Fitness: {self.fitness}"

# Fitness Evaluation Class using Multiprocessing
class PooledFitnessCompute(object):
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes_per_eval = 3000 # Max steps per trial
        self.num_trials = 16 # Number of trials per genome evaluation
        self.generation = 0
        # Note: env is created globally, multiprocessing might have issues if env
        # is not pickleable or if state needs to be independent per worker.
        # For SlimeVolley, this often works, but be cautious.
        # A safer approach involves initializing env within compute_fitness if issues arise.

    def compute_fitness(self, genome_tuple):
        """Evaluates a single genome."""
        genome_id, genome, config = genome_tuple
        try:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        except Exception as e:
            print(f"Error creating network for genome {genome_id}: {e}")
            return -float('inf') # Assign very low fitness if network creation fails

        trial_rewards = []
        for _ in range(self.num_trials):
            try:
                # Use the global env or re-initialize if needed:
                # with gym.make('SlimeVolley-v0') as trial_env:
                # observation, info = trial_env.reset() # Use newer reset API
                observation = env.reset() # Older API if needed
                if isinstance(observation, tuple): # Handle newer Gym API return
                     observation = observation[0]

                current_reward = 0
                step = 0
                while step < self.test_episodes_per_eval:
                    step += 1
                    action = net.activate(observation)
                    # Adapt based on environment's step return value
                    try:
                        # Newer Gym API
                        observation, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated
                    except ValueError:
                        # Older Gym API
                        observation, reward, done, info = env.step(action)

                    current_reward += reward
                    if done:
                        break
                trial_rewards.append(current_reward)

            except Exception as e:
                print(f"Error during simulation trial for genome {genome_id}: {e}")
                trial_rewards.append(-float('inf')) # Penalize errors during simulation

        # Calculate average reward across trials
        if not trial_rewards:
            return -float('inf') # Should not happen if num_trials > 0
        final_fitness = np.mean(trial_rewards)
        return final_fitness

    def evaluate_genomes(self, genomes, config):
        """Evaluates multiple genomes, potentially in parallel."""
        self.generation += 1
        print(f"\n--- Generation {self.generation} ---")
        t0 = time.time()

        jobs = []
        genomes_list = list(genomes) # Convert dict_items to list

        if self.num_workers < 2:
            # Serial evaluation
            print(f"Evaluating {len(genomes_list)} genomes serially...")
            for genome_id, genome in genomes_list:
                genome.fitness = self.compute_fitness((genome_id, genome, config))
        else:
            # Parallel evaluation
            print(f"Evaluating {len(genomes_list)} genomes using {self.num_workers} workers...")
            with multiprocessing.Pool(self.num_workers) as pool:
                # Prepare arguments for mapping
                tasks = [(gid, g, config) for gid, g in genomes_list]
                results = pool.map(self.compute_fitness, tasks)

                # Assign fitness back to genomes
                for i, (genome_id, genome) in enumerate(genomes_list):
                    genome.fitness = results[i]
                    # print(f"Genome {genome_id} fitness: {genome.fitness:.4f}") # Debug print

        eval_time = time.time() - t0
        print(f"Fitness evaluation time: {eval_time:.2f} seconds")


def run_experiment(config_file):
    """Runs the NEAT algorithm using the provided configuration file."""

    # Determine number of cores to use
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores.")

    # Load NEAT configuration
    print("Loading NEAT configuration...")
    config = neat.Config(SlimeGenome, # Use SlimeGenome or neat.DefaultGenome
                          neat.DefaultReproduction,
                          neat.DefaultSpeciesSet,
                          neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    print("Creating initial population...")
    p = neat.Population(config)

    # Add reporters to show progress in the terminal and save stats/checkpoints.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # Save checkpoints less frequently if saving best genome periodically
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_PREFIX)
    p.add_reporter(neat.Checkpointer(generation_interval=100, # Checkpoint interval
                                      time_interval_seconds=None, # Disable time interval
                                      filename_prefix=checkpoint_path))

    # Setup fitness evaluation
    pe = PooledFitnessCompute(num_cores)

    # Variables to track the best genome found so far
    overall_best_genome = None
    overall_best_fitness = -float('inf')
    start_time = time.time()

    # Main training loop
    try:
        while True: # Run indefinitely until solved or interrupted
            # Run one generation
            p.run(pe.evaluate_genomes, 1) # Run for 1 generation at a time

            # Get the best genome of the current generation
            current_best_genome = p.best_genome
            if current_best_genome is not None:
                 print(f"Generation {p.generation} best fitness: {current_best_genome.fitness:.4f}")

                 # Update overall best if current is better
                 if current_best_genome.fitness is not None and current_best_genome.fitness > overall_best_fitness:
                     overall_best_fitness = current_best_genome.fitness
                     overall_best_genome = current_best_genome
                     print(f"** New overall best fitness: {overall_best_fitness:.4f} in generation {p.generation} **")

            # Periodically save the best genome found SO FAR
            if overall_best_genome is not None and p.generation % SAVE_INTERVAL == 0:
                filename = os.path.join(OUTPUT_DIR, f"{BEST_GENOME_PREFIX}{p.generation}.pickle")
                print(f"Saving overall best genome (fitness {overall_best_fitness:.4f}) to {filename}")
                with open(filename, 'wb') as f:
                    pickle.dump(overall_best_genome, f)

                # Optionally save network visualization too
                vis_file_base = os.path.join(OUTPUT_DIR, f"{BEST_GENOME_PREFIX}{p.generation}")
                try:
                    visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net.gv")
                    visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net-pruned.gv", prune_unused=True)
                except Exception as e:
                    print(f"Warning: Could not generate network visualization. Is graphviz installed? Error: {e}")


            # Optional: Plot stats periodically
            if p.generation % 10 == 0: # Plot every 10 generations
                 try:
                     plot_filename = os.path.join(OUTPUT_DIR, "fitness_plot.svg")
                     visualize.plot_stats(stats, ylog=False, view=False, filename=plot_filename)
                     # visualize.plot_species(stats, view=False, filename=os.path.join(OUTPUT_DIR, "speciation_plot.svg"))
                 except Exception as e:
                     print(f"Warning: Could not generate plots. Error: {e}")


            # --- Simple Solved Condition (Example) ---
            # This is a basic example. Real "solved" criteria might be more complex
            # (e.g., average reward over multiple episodes > threshold).
            # We check the fitness of the *overall best* genome found so far.
            SOLVED_THRESHOLD = 5.0 # Example threshold for SlimeVolley (adjust as needed)
            if overall_best_fitness >= SOLVED_THRESHOLD:
                print(f"\nEnvironment considered SOLVED! Best fitness {overall_best_fitness:.4f} >= {SOLVED_THRESHOLD}")

                # Save the final absolute best genome
                if overall_best_genome:
                    final_filename = os.path.join(OUTPUT_DIR, FINAL_WINNER_FILENAME)
                    print(f"Saving final best genome to {final_filename}")
                    with open(final_filename, 'wb') as f:
                        pickle.dump(overall_best_genome, f)

                    # Save final visualizations
                    vis_file_base = os.path.join(OUTPUT_DIR, "final_winner")
                    try:
                        visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net.gv")
                        visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net-pruned.gv", prune_unused=True)
                    except Exception as e:
                        print(f"Warning: Could not generate final network visualization. Error: {e}")
                break # Exit the training loop

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save the best genome found so far upon interruption
        if overall_best_genome:
            interrupt_filename = os.path.join(OUTPUT_DIR, f"best_genome_interrupt_gen_{p.generation}.pickle")
            print(f"Saving current best genome (fitness {overall_best_fitness:.4f}) due to interruption to {interrupt_filename}")
            with open(interrupt_filename, 'wb') as f:
                pickle.dump(overall_best_genome, f)
    finally:
        # Clean up
        env.close()
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    # Ensure the config file exists
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: NEAT configuration file not found at {CONFIG_PATH}")
        print("Please ensure the 'task/config_slime' file exists relative to where you run the script.")
    else:
        run_experiment(CONFIG_PATH)

