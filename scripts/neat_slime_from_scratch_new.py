# Import necessary libraries
import neat
import numpy as np
import os
import pickle
import time
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Adjust path to NEAT
import utils.visualize as visualize # Keep for original visualization/stats
import slimevolleygym
import gym
import gym.wrappers
import multiprocessing
import logging

# --- Add NetworkX and Matplotlib imports ---
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    _NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: networkx or matplotlib not found. NetworkX plotting disabled.")
    print("Install with: pip install networkx matplotlib")
    _NETWORKX_AVAILABLE = False
# --- End imports ---


# --- Configuration ---
CONFIG_PATH = os.path.join('task', 'config_slime')
SAVE_INTERVAL = 500 # Save genome and plots every 50 generations
OUTPUT_DIR = f'results/run_1'
BEST_GENOME_PREFIX = 'best_genome_gen_'
FINAL_WINNER_FILENAME = 'final_winner.pickle'
CHECKPOINT_PREFIX = 'neat-checkpoint-'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Environment Setup (logging, warnings, gym) ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
import warnings
from pkg_resources import PkgResourcesDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

env = gym.make('SlimeVolley-v0')
# print(f"Environment Observation Space: {env.observation_space}")
# print(f"Environment Action Space: {env.action_space}")

# --- NEAT Setup (Genome Class, Fitness Compute Class) ---
class SlimeGenome(neat.DefaultGenome): # Assuming this is still used
    def __init__(self, key):
        super().__init__(key)
    def __str__(self):
        return f"SlimeGenome {self.key}, Fitness: {self.fitness}"

class PooledFitnessCompute(object): # Assuming definition from previous steps
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.test_episodes_per_eval = 3000
        self.num_trials = 16
        self.generation = 0
    def compute_fitness(self, genome_tuple):
        genome_id, genome, config = genome_tuple
        try: net = neat.nn.FeedForwardNetwork.create(genome, config)
        except Exception: return -float('inf')
        trial_rewards = []
        for _ in range(self.num_trials):
            try:
                observation = env.reset() # Assumes older gym API
                current_reward = 0; step = 0
                while step < self.test_episodes_per_eval:
                    step += 1
                    action = net.activate(observation)
                    observation, reward, done, info = env.step(action) # Assumes older gym API
                    current_reward += reward
                    if done: break
                trial_rewards.append(current_reward)
            except Exception: trial_rewards.append(-float('inf'))
        if not trial_rewards: return -float('inf')
        return np.mean(trial_rewards)
    def evaluate_genomes(self, genomes, config):
        self.generation += 1
        print(f"\n--- Generation {self.generation} ---")
        t0 = time.time()
        genomes_list = list(genomes)
        if self.num_workers < 2:
            print(f"Evaluating {len(genomes_list)} genomes serially...")
            for genome_id, genome in genomes_list:
                genome.fitness = self.compute_fitness((genome_id, genome, config))
        else:
            print(f"Evaluating {len(genomes_list)} genomes using {self.num_workers} workers...")
            with multiprocessing.Pool(self.num_workers) as pool:
                tasks = [(gid, g, config) for gid, g in genomes_list]
                results = pool.map(self.compute_fitness, tasks)
                for i, (genome_id, genome) in enumerate(genomes_list):
                    genome.fitness = results[i]
        eval_time = time.time() - t0
        print(f"Fitness evaluation time: {eval_time:.2f} seconds")


# === START OF NEW NETWORKX PLOTTING FUNCTION ===
# Added 'generation' parameter
def plot_networkx(config, genome, generation, filename="neat_networkx.png", view=False):
    """
    Generates a plot of the NEAT network using NetworkX and Matplotlib.
    Args:
        config: The NEAT configuration object.
        genome: The NEAT genome object to plot.
        generation: The current generation number (for the title). # <-- Added Arg
        filename: The path to save the plot image file (e.g., .png, .svg).
        view: (Not recommended for training loops) If True, try to display the plot window.
    """
    if not _NETWORKX_AVAILABLE:
         print("Skipping NetworkX plot: networkx or matplotlib not installed.")
         return

    print(f"Generating NetworkX plot and saving to {filename}...")
    nx_graph = nx.DiGraph()
    node_colors, node_labels, node_sizes = {}, {}, {}
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys

    # Add nodes to graph and set properties
    for k in input_keys:
        nx_graph.add_node(k)
        node_colors[k] = 'lightblue'; node_labels[k] = f"I{k}"; node_sizes[k] = 700
    for k in output_keys:
        nx_graph.add_node(k)
        node_colors[k] = 'lightgreen'; node_labels[k] = f"O{k}"; node_sizes[k] = 700
    for k in genome.nodes:
        if k not in input_keys and k not in output_keys:
            nx_graph.add_node(k)
            node_colors[k] = 'lightgray'; node_labels[k] = str(k); node_sizes[k] = 500

    # Add edges to graph and set properties
    edge_colors, edge_widths = [], []
    for cg in genome.connections.values():
        if cg.enabled:
            inp, outp = cg.key
            if inp in nx_graph and outp in nx_graph:
                 nx_graph.add_edge(inp, outp)
                 edge_colors.append('green' if cg.weight > 0 else 'red')
                 edge_widths.append(1 + abs(cg.weight) * 1.5)
            else:
                 print(f"Warning: Skipping connection {cg.key} - node missing from graph.")

    # Create Matplotlib figure
    fig = plt.figure(figsize=(15, 12))
    try:
        # --- Layout ---
        pos = nx.kamada_kawai_layout(nx_graph) # Calculate node positions

        # --- Drawing ---
        nx.draw_networkx_nodes(nx_graph, pos,
                               node_color=[node_colors.get(n, 'gray') for n in nx_graph.nodes()],
                               node_size=[node_sizes.get(n, 300) for n in nx_graph.nodes()])
        nx.draw_networkx_edges(nx_graph, pos,
                               edge_color=edge_colors,
                               width=edge_widths,
                               arrowstyle='-|>', arrowsize=12, alpha=0.6)
        nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)

        # --- Final Plot Touches ---
        # Use the 'generation' parameter passed to the function
        plt.title(f"NEAT Network (Gen: {generation}, Key: {genome.key}, Fitness: {genome.fitness:.4f})", fontsize=14)
        plt.xticks([]); plt.yticks([]) # Hide axes
        plt.tight_layout()

        # --- Saving ---
        plt.savefig(filename, format=filename.split('.')[-1], dpi=150) # Save figure
        print(f"NetworkX plot saved successfully to {filename}")

    except Exception as plot_err:
        print(f"Error during NetworkX plotting or saving: {plot_err}")
    finally:
        plt.close(fig) # Close the figure to free memory, important in loops!

# === END OF NEW NETWORKX PLOTTING FUNCTION ===


def run_experiment(config_file):
    """Runs the NEAT algorithm using the provided configuration file."""
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores.")
    config = neat.Config(SlimeGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT_PREFIX)
    p.add_reporter(neat.Checkpointer(generation_interval=100, time_interval_seconds=None,
                                      filename_prefix=checkpoint_path))
    pe = PooledFitnessCompute(num_cores)
    overall_best_genome = None
    overall_best_fitness = -float('inf')
    start_time = time.time()

    try:
        while True: # Main training loop
            # --- Run Generation ---
            # genome.generation = p.generation # <--- DELETED THIS LINE
            p.run(pe.evaluate_genomes, 1)

            # --- Update Best ---
            current_best_genome = p.best_genome
            if current_best_genome is not None:
                 print(f"Generation {p.generation} best fitness: {current_best_genome.fitness:.4f}")
                 if current_best_genome.fitness is not None and current_best_genome.fitness > overall_best_fitness:
                     overall_best_fitness = current_best_genome.fitness
                     overall_best_genome = current_best_genome
                     print(f"** New overall best fitness: {overall_best_fitness:.4f} in generation {p.generation} **")

            # === Periodic Saving and Visualization Block ===
            if overall_best_genome is not None and p.generation > 0 and p.generation % SAVE_INTERVAL == 0:
                # Base filename for this generation's artifacts
                vis_file_base = os.path.join(OUTPUT_DIR, f"{BEST_GENOME_PREFIX}{p.generation}")

                # --- Save Best Genome Object ---
                pickle_filename = vis_file_base + ".pickle"
                print(f"Saving overall best genome (fitness {overall_best_fitness:.4f}) to {pickle_filename}")
                with open(pickle_filename, 'wb') as f:
                    pickle.dump(overall_best_genome, f)

                # --- Visualize and Save Network Graphs (Graphviz .gv files) ---
                print(f"Saving Graphviz network visualization to {vis_file_base}-net.gv and -pruned.gv")
                try:
                    visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net.gv")
                    visualize.draw_net(config, overall_best_genome, view=False, filename=vis_file_base + "-net-pruned.gv", prune_unused=True)
                    print(f"Graphviz files saved. Use 'dot' command to view (e.g., dot -Tpng {vis_file_base}-net.gv -o output.png)")
                except Exception as e:
                    print(f"Warning: Could not generate Graphviz network visualization. Is Graphviz installed? Error: {e}")

                # --- Visualize and Save Network Plot (NetworkX/Matplotlib PNG) ---
                nx_plot_filename = vis_file_base + "-net.png" # Save as PNG
                # Call the plotting function, passing p.generation
                plot_networkx(config, overall_best_genome, p.generation, filename=nx_plot_filename) # <-- Pass p.generation
                # === End of Periodic Saving / Visualization ===

            # --- Plot Stats ---
            if p.generation > 0 and p.generation % 10 == 0:
                 try:
                     plot_filename = os.path.join(OUTPUT_DIR, "fitness_plot.svg")
                     visualize.plot_stats(stats, ylog=False, view=False, filename=plot_filename)
                 except Exception as e:
                     print(f"Warning: Could not generate fitness plots. Error: {e}")

            # --- Check Solved Condition ---
            SOLVED_THRESHOLD = 5.0 # Example
            if overall_best_fitness >= SOLVED_THRESHOLD:
                print(f"\nEnvironment considered SOLVED! Best fitness {overall_best_fitness:.4f} >= {SOLVED_THRESHOLD}")
                # Save final winner artifacts (pickle, gv, png)
                if overall_best_genome:
                    final_vis_base = os.path.join(OUTPUT_DIR, "final_winner")
                    final_pickle = final_vis_base + ".pickle"
                    print(f"Saving final best genome to {final_pickle}")
                    with open(final_pickle, 'wb') as f: pickle.dump(overall_best_genome, f)
                    try: # Save final gv files
                        visualize.draw_net(config, overall_best_genome, view=False, filename=final_vis_base + "-net.gv")
                        visualize.draw_net(config, overall_best_genome, view=False, filename=final_vis_base + "-net-pruned.gv", prune_unused=True)
                    except Exception as e: print(f"Warning: Could not save final Graphviz files. Error: {e}")
                    # Save final networkx plot
                    plot_networkx(config, overall_best_genome, p.generation, filename=final_vis_base + "-net.png") # <-- Pass p.generation
                break # Exit loop

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save artifacts on interrupt
        if overall_best_genome:
            interrupt_vis_base = os.path.join(OUTPUT_DIR, f"best_genome_interrupt_gen_{p.generation}")
            interrupt_pickle = interrupt_vis_base + ".pickle"
            print(f"Saving current best genome (fitness {overall_best_fitness:.4f}) to {interrupt_pickle}")
            with open(interrupt_pickle, 'wb') as f: pickle.dump(overall_best_genome, f)
            try: # Save gv on interrupt
                visualize.draw_net(config, overall_best_genome, view=False, filename=interrupt_vis_base + "-net.gv")
                visualize.draw_net(config, overall_best_genome, view=False, filename=interrupt_vis_base + "-net-pruned.gv", prune_unused=True)
            except Exception as e: print(f"Warning: Could not save interrupt Graphviz files. Error: {e}")
            # Save networkx plot on interrupt
            plot_networkx(config, overall_best_genome, p.generation, filename=interrupt_vis_base + "-net.png") # <-- Pass p.generation

    finally:
        # Cleanup
        env.close()
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        print(f"Results saved in: {OUTPUT_DIR}")

# --- Main execution block ---
if __name__ == '__main__':
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: NEAT configuration file not found at {CONFIG_PATH}")
    else:
        run_experiment(CONFIG_PATH)

