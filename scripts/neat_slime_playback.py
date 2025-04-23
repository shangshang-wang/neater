# Import necessary libraries
import neat
import os
import pickle
import slimevolleygym
import gym
import time
import argparse
import numpy as np

# Attempt to import imageio for manual recording
try:
    import imageio
except ImportError:
    print("---------------------------------------------------------")
    print("Warning: 'imageio' library not found.")
    print("Video recording will not be available.")
    print("Please install it: pip install imageio imageio-ffmpeg")
    print("---------------------------------------------------------")
    imageio = None # Set to None if import fails

# Attempt to import pyglet to check instance type (optional but helpful)
# Pyglet is often used by older Gym versions for rendering.
try:
    import pyglet
except ImportError:
    pyglet = None # Set to None if pyglet is not installed or needed

# --- Configuration ---
CONFIG_PATH = os.path.join('task', 'config_slime') # Path to NEAT config
DEFAULT_VIDEO_DIR = 'videos' # Default save directory for videos
DEFAULT_FPS = 30 # Default frames per second for recorded video

# --- NEAT Genome Class ---
# Ensure this matches the Genome class used during training
class SlimeGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
    def __str__(self):
        return f"SlimeGenome {self.key}, Fitness: {self.fitness}"

# --- Helper Functions ---
def load_genome(filename):
    """Loads a genome object from a pickle file."""
    print(f"Loading genome from: {filename}")
    try:
        with open(filename, 'rb') as f:
            genome = pickle.load(f)
        return genome
    except FileNotFoundError:
        print(f"Error: Genome file not found at '{filename}'")
        return None
    except Exception as e:
        print(f"Error loading genome file: {e}")
        return None

def load_config(config_file):
    """Loads the NEAT configuration."""
    print(f"Loading NEAT configuration from: {config_file}")
    try:
        # Ensure SlimeGenome (or the correct Genome class) is used here
        config = neat.Config(SlimeGenome,
                             neat.DefaultReproduction,
                             neat.DefaultSpeciesSet,
                             neat.DefaultStagnation,
                             config_file)
        return config
    except Exception as e:
        print(f"Error loading NEAT config: {e}")
        return None

def extract_numpy_frame(frame_data):
    """
    Attempts to extract a NumPy array from the data returned by env.render(mode='rgb_array').
    Handles direct NumPy arrays and common Pyglet ImageData objects.
    May need adjustments based on the actual object type returned by the environment.
    """
    if isinstance(frame_data, np.ndarray):
        # If it's already a NumPy array, return it directly
        return frame_data
    elif pyglet and hasattr(pyglet, 'image') and isinstance(frame_data, pyglet.image.ImageData):
        # Handle pyglet.image.ImageData if pyglet is available and type matches
        try:
            # Get image properties
            format_str = frame_data.format
            pitch = frame_data.pitch # Bytes per row
            width = frame_data.width
            height = frame_data.height
            channels = len(format_str) # e.g., 3 for 'RGB', 4 for 'RGBA'

            # Get raw byte data from the pyglet image object
            bytes_data = frame_data.get_data(format_str, pitch)

            # Convert raw bytes to a 1D NumPy array
            np_array = np.frombuffer(bytes_data, dtype=np.uint8)

            # Reshape the array. Handle potential row padding (pitch != width * channels).
            if pitch == width * channels:
                 # No padding, direct reshape
                 np_array = np_array.reshape(height, width, channels)
            elif pitch > width * channels:
                 # Pitch includes padding, reshape to height x pitch/channels x channels
                 # then slice off the padding pixels from each row.
                 np_array = np_array.reshape(height, pitch // channels, channels)[:, :width, :]
                 print(f"Note: Handled image pitch ({pitch}) != width*channels ({width}*{channels}).")
            else:
                 # This case shouldn't normally happen if pitch is correct
                 print(f"Warning: Image pitch ({pitch}) < width*channels ({width}*{channels}). Frame extraction might be incorrect.")
                 # Attempt direct reshape, may fail or be distorted
                 np_array = np_array.reshape(height, width, channels)

            # Pyglet often uses OpenGL's coordinate system (origin at bottom-left),
            # resulting in images that are vertically flipped compared to standard image formats.
            # Flip it vertically (up-down) to correct the orientation.
            return np.flipud(np_array)

        except Exception as e:
            print(f"Error extracting frame from Pyglet object: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for pyglet errors
            return None
    elif hasattr(frame_data, 'data') and isinstance(getattr(frame_data, 'data', None), np.ndarray):
         # Handle other potential generic objects that might have a .data attribute
         # which holds the numpy array.
         print("[Debug] Extracting frame via .data attribute")
         return frame_data.data
    else:
        # If the object type is unknown or cannot be handled
        print(f"Warning: Unexpected frame data type encountered: {type(frame_data)}. Cannot extract NumPy array.")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Load a NEAT genome and record (manually) or watch SlimeVolley gameplay.")
    parser.add_argument("genome_file", help="Path to the saved genome (.pickle file).")
    parser.add_argument("--watch-only", action="store_true",
                        help="Watch the agent play live instead of recording a video.")
    parser.add_argument("-o", "--output-dir", default=DEFAULT_VIDEO_DIR,
                        help=f"Directory to save the recorded video(s) (default: {DEFAULT_VIDEO_DIR}). Only used if --watch-only is NOT specified.")
    parser.add_argument("-n", "--num-episodes", type=int, default=3,
                        help="Number of episodes to record or watch (default: 3).")
    parser.add_argument("-c", "--config", default=CONFIG_PATH,
                        help=f"Path to the NEAT config file (default: {CONFIG_PATH}).")
    parser.add_argument("--sleep", type=float, default=0.01,
                        help="Seconds to sleep between steps (slows down playback, default: 0.01).")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS,
                        help=f"FPS for recorded video (default: {DEFAULT_FPS}). Only used if recording.")
    args = parser.parse_args()

    # --- Load Genome and Config ---
    genome = load_genome(args.genome_file)
    if genome is None: exit(1) # Exit if genome loading failed
    config = load_config(args.config)
    if config is None: exit(1) # Exit if config loading failed

    # --- Create Network ---
    print("Creating neural network from genome...")
    try:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    except Exception as e:
        print(f"Error creating network: {e}")
        exit(1)

    # --- Setup Environment ---
    print("Setting up environment...")
    try:
        # Create the environment. No special render_mode needed at init for manual capture/display.
        env = gym.make('SlimeVolley-v0')
    except Exception as e:
        print(f"Error creating environment: {e}")
        exit(1)

    # --- Display Mode ---
    print(f"\nStarting live playback for {args.num_episodes} episodes...")

    # --- Generate Base Video Filename ---
    video_name_base = os.path.splitext(os.path.basename(args.genome_file))[0]

    # --- Simulation Loop ---
    for i in range(args.num_episodes):
        print(f"\n--- Episode {i+1} ---")
        frames = [] # Initialize list to store frames for this episode if recording
        try:
            # Reset environment using the older Gym API (returns observation only)
            observation = env.reset()
            done = False
            total_reward = 0
            step = 0
            first_frame_info = True # Flag to print debug info only once per episode

            while not done:
                # Get action from the loaded neural network
                action = net.activate(observation)

                # Step the environment using the older Gym API (returns 4 values)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                step += 1

                # --- Handle Rendering/Recording ---
                try:
                    env.render()
                except Exception as render_err:
                    # Log warning if rendering to screen fails, but continue
                    print(f"Warning: Screen rendering call failed - {render_err}", end='\r')
                    pass

                if args.sleep > 0:
                    time.sleep(args.sleep)

            # --- Episode End ---
            print(f"\nEpisode {i+1} finished after {step} steps.")
            print(f"Total reward: {total_reward}")

        except Exception as e:
            print(f"\nAn error occurred during episode {i+1}: {e}")
            break

    print("\nClosing environment...")
    env.close()
    print("Playback finished.")
