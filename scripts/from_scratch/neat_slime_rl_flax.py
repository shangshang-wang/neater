import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gym
import slimevolleygym

from functools import partial
import hydra
import flax
from flax import serialization
from flax.training import train_state, checkpoints
from flax import linen as nn
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax
import os
import csv
from tqdm import tqdm
import time


def convert_params_to_json_serializable(params):
    if isinstance(params, (jnp.ndarray, np.ndarray)):
        return params.tolist()
    elif isinstance(params, (dict, flax.core.FrozenDict)):
        serializable_dict = {}
        for k, v in params.items():
            serializable_dict[k] = convert_params_to_json_serializable(v)
        return serializable_dict
    elif isinstance(params, (list, tuple)):
        return [convert_params_to_json_serializable(item) for item in params]
    elif isinstance(params, (int, float, str, bool, type(None))):
        return params
    else:
        print(f"Warning: Converting unrecognized type {type(params)} to string for JSON.")
        return str(params)

def draw_graph(trained_model_state, activations_list, cfg):
    G = nx.DiGraph()

    if not hasattr(trained_model_state, 'params') or not trained_model_state.params:
         print("Warning: Model parameters missing or empty in TrainState. Cannot draw graph.")
         plt.figure(figsize=(12, 8))
         plt.title("Graph Generation Skipped (Missing/Empty Params)")
         plt.close()
         return

    param_dict_maybe_nested = trained_model_state.params
    param_dict = param_dict_maybe_nested.get('params', param_dict_maybe_nested)

    if not isinstance(param_dict, (dict, flax.core.FrozenDict)) or not param_dict:
        print(f"Warning: Parameter dictionary is empty or not a valid dictionary type ({type(param_dict)}). Cannot draw graph.")
        plt.figure(figsize=(12, 8))
        plt.title("Graph Generation Skipped (Empty/Invalid Params Dict)")
        plt.close()
        return

    num_elements = cfg.network.num_inputs
    start_nodes = [f'obs_node_{i + 1}' for i in range(min(num_elements, 6))]
    if num_elements > 6:
        start_nodes = start_nodes[:3] + ['obs_ellipsis'] + start_nodes[-3:]

    for i, node in enumerate(start_nodes):
        if node == 'obs_ellipsis':
            y_pos = 0
            G.add_node(node, pos=(0, y_pos), shape='dot')
            G.add_node(f"{node}_dot2", pos=(0, y_pos - 0.12), shape='dot')
            G.add_node(f"{node}_dot3", pos=(0, y_pos + 0.12), shape='dot')
        else:
            adj_idx = i if num_elements <= 6 or i < 3 else i - 1
            total_vis_nodes = min(num_elements, 7)
            y_pos = adj_idx - (total_vis_nodes - 1) / 2.0
            G.add_node(node, pos=(0, y_pos))

    prev_layer_node_names_for_edges = [n for n in start_nodes if 'ellipsis' not in n]
    layer_distance = 0.5
    edge_weights_dict = {}
    layer_items = list(param_dict.items())

    for layer_index, (layer, value) in enumerate(layer_items):
        if 'kernel' not in value: continue
        kernel_shape = value['kernel'].shape
        if len(kernel_shape) != 2: continue

        num_prev_nodes_actual = kernel_shape[0]
        num_nodes_actual = kernel_shape[1]
        num_nodes_vis = min(num_nodes_actual, 6)
        use_ellipsis_current = num_nodes_actual > 6
        vis_indices_current = list(range(num_nodes_vis))
        logical_vis_count = num_nodes_actual
        if use_ellipsis_current:
            vis_indices_current = list(range(3)) + [-1] + list(range(num_nodes_actual - 3, num_nodes_actual))
            logical_vis_count = 7

        current_layer_node_names_for_edges = []
        current_layer_nodes_vis = []

        for node_i_vis in range(logical_vis_count):
            is_ellipsis = use_ellipsis_current and node_i_vis == 3
            node_actual_idx = vis_indices_current[node_i_vis] if not is_ellipsis else -1
            current_node_name = f"{layer}_node_{node_actual_idx}" if not is_ellipsis else f"{layer}_ellipsis"
            y_pos = node_i_vis - (logical_vis_count - 1) / 2.0
            current_layer_nodes_vis.append(current_node_name)

            if is_ellipsis:
                G.add_node(current_node_name, pos=(layer_distance, y_pos), shape='dot')
                G.add_node(f"{current_node_name}_dot2", pos=(layer_distance, y_pos - 0.12), shape='dot')
                G.add_node(f"{current_node_name}_dot3", pos=(layer_distance, y_pos + 0.12), shape='dot')
            else:
                G.add_node(current_node_name, pos=(layer_distance, y_pos))
                current_layer_node_names_for_edges.append(current_node_name)

                for prev_node_name in prev_layer_node_names_for_edges:
                     try:
                         parts = prev_node_name.split('_')
                         if parts[-2] == 'node':
                             prev_node_actual_idx = int(parts[-1])
                         elif parts[0] == 'obs' and parts[1] == 'node':
                              prev_node_actual_idx = int(parts[2]) - 1
                         else: continue
                     except (IndexError, ValueError):
                         continue

                     if 0 <= prev_node_actual_idx < num_prev_nodes_actual and 0 <= node_actual_idx < num_nodes_actual:
                         weight_value = value['kernel'][prev_node_actual_idx][node_actual_idx]
                         G.add_edge(prev_node_name, current_node_name, weight=weight_value)
                         edge_weights_dict[(prev_node_name, current_node_name)] = f"{weight_value:.2f}"

        prev_layer_node_names_for_edges = current_layer_node_names_for_edges
        layer_distance += 0.5

    plt.figure(figsize=(15, 10))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
         print("Warning: Graph position attribute is empty. Skipping draw.")
         plt.title("Graph Generation Skipped (Empty Positions)")
         plt.close()
         return

    y_coords = [p[1] for p in pos.values()]
    max_y = max(y_coords) if y_coords else 0
    height_text = max_y + 2.0
    activation_text_height = height_text - 0.5

    layer_items_for_text = list(param_dict.items())
    count = 1
    for i, (layer, value) in enumerate(layer_items_for_text):
         if 'kernel' not in value: continue
         num_nodes_actual = value['kernel'].shape[1]
         current_layer_x = 0.5 * (i + 1)

         is_output_layer = (i == len(layer_items_for_text) - 1)

         if not is_output_layer:
            plt.text(current_layer_x, height_text, f"HL {count}", fontsize=11, color='black', ha='center', va='center')
            plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')
            if i < len(activations_list):
                activation_name = activations_list[i].__name__
                plt.text(current_layer_x, activation_text_height, f"act: {activation_name}", fontsize=9, color='blue', ha='center', va='center')
            count += 1
         else:
             plt.text(current_layer_x, height_text, f"Output Layer", fontsize=11, color='black', ha='center', va='center')
             plt.text(current_layer_x, height_text - 0.25, f"actions: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')

    input_layer_x = 0
    plt.text(input_layer_x, height_text, f"Input Layer", fontsize=11, color='black', ha='center', va='center')
    plt.text(input_layer_x, height_text - 0.25, f"observations: {cfg.network.num_inputs}", fontsize=9, color='black', ha='center', va='center')

    edge_weights_vals = [abs(d.get('weight', 0)) for u, v, d in G.edges(data=True)]
    max_edge_width = max(edge_weights_vals) if edge_weights_vals else 1.0
    edge_widths_normalized = [(abs(w) / max_edge_width * 2.5 + 0.2) if max_edge_width > 0 else 0.2 for w in edge_weights_vals]

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6',
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') != 'dot'])
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='grey',
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') == 'dot'])
    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized, alpha=0.5, node_size=400)

    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_weights_dict, font_size=7, font_color='red',
        label_pos=0.3, bbox=dict(alpha=0)
    )

    plt.tight_layout()
    plt.axis('off')


POSSIBLE_ACTIVATIONS = {
    'relu': nn.relu, 'sigmoid': nn.sigmoid, 'tanh': nn.tanh,
    'leaky_relu': nn.leaky_relu, 'swish': nn.swish, 'gelu': nn.gelu
}
POSSIBLE_ACTIVATION_FNS = list(POSSIBLE_ACTIVATIONS.values())
POSSIBLE_ACTIVATION_NAMES = list(POSSIBLE_ACTIVATIONS.keys())

class GenomeClassifier(nn.Module):
    layer_definitions: list
    activation_fns: list

    @nn.compact
    def __call__(self, x):
        if len(self.layer_definitions) != len(self.activation_fns) + 1:
             raise ValueError(f"Mismatch layers ({len(self.layer_definitions)}) vs activations ({len(self.activation_fns)})")
        for i in range(len(self.activation_fns)):
            x = self.layer_definitions[i](x)
            x = self.activation_fns[i](x)
        x = self.layer_definitions[-1](x)
        return x

def create_layers(rng, layer_sizes, num_output, current_activations=None):
    layers = []
    activations = []
    num_hidden_layers = len(layer_sizes)
    if current_activations is not None and len(current_activations) != num_hidden_layers:
        print(f"Warning: Activation list length mismatch. Re-initializing.")
        current_activations = None

    for i, hidden_size in enumerate(layer_sizes):
        layers.append(nn.Dense(features=hidden_size))
        if current_activations is None:
            default_activation = nn.tanh
            activations.append(default_activation)
        else:
            activations.append(current_activations[i])
    layers.append(nn.Dense(features=num_output))
    return layers, activations

def copy_params_and_create_model(rng, layer_sizes, current_activations, num_output, num_inputs, source_params=None):
    rng, layer_rng, init_rng, input_rng = jax.random.split(rng, 4)
    layers, activations = create_layers(layer_rng, layer_sizes, num_output, current_activations)
    new_model = GenomeClassifier(layer_definitions=layers, activation_fns=activations)

    dummy_input = jax.random.normal(input_rng, (num_inputs,))
    target_params = new_model.init(init_rng, dummy_input)['params']
    new_params = target_params

    if source_params is None:
        return new_model, {'params': new_params}

    new_params_mutable = flax.core.unfreeze(new_params)
    if isinstance(source_params, flax.core.FrozenDict):
        source_params_dict = source_params.unfreeze()
    else:
        source_params_dict = source_params.get('params', source_params)
        if isinstance(source_params_dict, flax.core.FrozenDict):
             source_params_dict = source_params_dict.unfreeze()

    source_params_layers = list(source_params_dict.items())
    target_params_layers = list(new_params_mutable.items())
    num_layers_to_copy = min(len(source_params_layers), len(target_params_layers))

    for i in range(num_layers_to_copy):
        source_layer_name, source_layer_values = source_params_layers[i]
        target_layer_name, target_layer_values = target_params_layers[i]
        if 'kernel' in source_layer_values and 'kernel' in target_layer_values:
            source_kernel = source_layer_values['kernel']
            target_kernel = target_layer_values['kernel']
            copy_rows = min(source_kernel.shape[0], target_kernel.shape[0])
            copy_cols = min(source_kernel.shape[1], target_kernel.shape[1])
            updated_kernel = target_kernel.at[:copy_rows, :copy_cols].set(source_kernel[:copy_rows, :copy_cols])
            new_params_mutable[target_layer_name]['kernel'] = updated_kernel
        if 'bias' in source_layer_values and 'bias' in target_layer_values:
            source_bias = source_layer_values['bias']
            target_bias = target_layer_values['bias']
            copy_len = min(len(source_bias), len(target_bias))
            updated_bias = target_bias.at[:copy_len].set(source_bias[:copy_len])
            new_params_mutable[target_layer_name]['bias'] = updated_bias
    return new_model, {'params': flax.core.freeze(new_params_mutable)}

def add_new_layer(rng, layer_sizes, current_activations, source_params, cfg):
    rng, insert_rng, act_rng = jax.random.split(rng, 3)
    insert_pos = jax.random.randint(insert_rng, (1,), 0, len(layer_sizes) + 1).item()
    new_layer_size = 1
    new_layer_sizes = layer_sizes[:insert_pos] + [new_layer_size] + layer_sizes[insert_pos:]
    num_possible = len(POSSIBLE_ACTIVATION_FNS)
    indices = jnp.arange(num_possible)
    chosen_index = jax.random.choice(act_rng, indices)
    new_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
    new_activations = current_activations[:insert_pos] + [new_activation] + current_activations[insert_pos:]
    print(f"  New Layer Sizes: {new_layer_sizes}")
    print(f"  New Activations: {[a.__name__ for a in new_activations]}")
    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations

def add_new_node(rng, layer_sizes, current_activations, source_params, cfg):
    if not layer_sizes:
        print("  Skipping Add Node: No hidden layers exist.")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.randint(choice_rng, (1,), 0, len(layer_sizes)).item()
    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] += 1
    print(f"  Incremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")
    new_activations = current_activations
    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations

def remove_node(rng, layer_sizes, current_activations, source_params, cfg):
    eligible_layers_indices = [i for i, size in enumerate(layer_sizes) if size > 1]
    if not eligible_layers_indices:
        print("  Skipping Remove Node: No eligible layers found.")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.choice(choice_rng, jnp.array(eligible_layers_indices)).item()
    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] -= 1
    print(f"  Decremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")
    new_activations = current_activations
    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations

def remove_layer(rng, layer_sizes, current_activations, source_params, cfg):
    if not layer_sizes:
        print("  Skipping Remove Layer: No hidden layers to remove.")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_remove_idx = jax.random.randint(choice_rng, (1,), 0, len(layer_sizes)).item()
    new_layer_sizes = layer_sizes[:layer_to_remove_idx] + layer_sizes[layer_to_remove_idx+1:]
    new_activations = current_activations[:layer_to_remove_idx] + current_activations[layer_to_remove_idx+1:]
    print(f"  Removed layer at index {layer_to_remove_idx}. New Sizes: {new_layer_sizes}")
    print(f"  New Activations: {[a.__name__ for a in new_activations]}")
    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations

def mutate_activation(rng, layer_sizes, current_activations, source_params, cfg):
    if not current_activations:
        print("  Skipping Mutate Activation: No hidden layers/activations exist.")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations
    rng, choice_rng, act_rng = jax.random.split(rng, 3)
    layer_to_mutate_idx = jax.random.randint(choice_rng, (1,), 0, len(current_activations)).item()
    current_activation_fn = current_activations[layer_to_mutate_idx]
    possible_new_activations = [fn for fn in POSSIBLE_ACTIVATION_FNS if fn != current_activation_fn]
    if not possible_new_activations:
        new_activation_fn = current_activation_fn
    else:
        possible_indices = [i for i, fn in enumerate(POSSIBLE_ACTIVATION_FNS) if fn in possible_new_activations]
        chosen_relative_index = jax.random.choice(act_rng, jnp.arange(len(possible_indices)))
        absolute_index = possible_indices[chosen_relative_index]
        new_activation_fn = POSSIBLE_ACTIVATION_FNS[absolute_index]
    new_activations = current_activations.copy()
    new_activations[layer_to_mutate_idx] = new_activation_fn
    print(f"  Mutated activation for layer {layer_to_mutate_idx} to {new_activation_fn.__name__}")
    print(f"  New Activations: {[a.__name__ for a in new_activations]}")
    return copy_params_and_create_model(rng, layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), layer_sizes, new_activations


@partial(jax.jit, static_argnames=['apply_fn'])
def select_action(apply_fn, params, obs, rng):
    logits = apply_fn({'params': params}, obs)
    action = jax.random.categorical(rng, logits)
    log_prob = jax.nn.log_softmax(logits)[action]
    return action, log_prob

def calculate_discounted_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = jnp.array(returns)
    return returns

def calculate_policy_loss(params, apply_fn, observations, actions, log_probs_old, returns):
    logits = vmap(apply_fn, in_axes=({'params': None}, 0))({'params': params}, observations)
    all_log_probs = jax.nn.log_softmax(logits)
    log_probs_taken = jnp.take_along_axis(all_log_probs, actions[:, None], axis=1).squeeze()
    loss = -jnp.mean(returns * log_probs_taken)
    return loss

# @partial(jax.jit, static_argnames=['apply_fn', 'optimizer'])
@jax.jit
def train_step_rl(state, observations, actions, log_probs_old, returns):
    grad_fn = jax.value_and_grad(calculate_policy_loss, argnums=0)
    loss, grads = grad_fn(state.params['params'], state.apply_fn, observations, actions, log_probs_old, returns)
    return state, loss


def run_episode(env, state, rng):
    observations, actions, rewards, log_probs = [], [], [], []
    obs = env.reset()
    obs = jnp.asarray(obs, dtype=jnp.float32)
    terminated = False
    total_reward = 0.0

    while not terminated:
        rng, act_rng = jax.random.split(rng)
        obs_input = obs

        # Get discrete action (0-7) and its log_prob from policy
        action_discrete, log_prob = select_action(state.apply_fn, state.params['params'], obs_input, act_rng)
        action_discrete_np = np.array(action_discrete) # scalar numpy int

        # Convert discrete action (0-7) to MultiBinary(3) format
        # The standard mapping often uses the binary representation.
        # Index `i` corresponds to the binary representation of `i` padded to 3 digits.
        # We need to map this to [FORWARD, BACKWARD, JUMP]
        # Example: 5 (binary 101) -> [1, 0, 1] -> Forward=1, Backward=0, Jump=1
        # Use numpy's right shift and bitwise AND for an efficient conversion:
        # ((action_discrete_np >> np.arange(3)) & 1) produces [J, B, F] order
        # We need [F, B, J], so we reverse the result.
        action_multibinary = ((action_discrete_np >> np.arange(3)) & 1)[::-1].astype(env.action_space.dtype)

        next_obs, reward, terminated, info = env.step(action_multibinary)
        next_obs = jnp.asarray(next_obs, dtype=jnp.float32)

        observations.append(obs_input)
        actions.append(action_discrete)
        rewards.append(float(reward))
        log_probs.append(log_prob)

        obs = next_obs
        total_reward += float(reward)

    return observations, actions, rewards, log_probs, total_reward, rng

def evaluate_policy(env, state, rng, num_eval_episodes=10):
    total_rewards = []
    for _ in range(num_eval_episodes):
        _, _, _, _, episode_reward, rng = run_episode(env, state, rng)
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), rng


@hydra.main(version_base=None, config_path="../../task", config_name="config_slime")
def main(cfg):
    results_dir = "assets/results/from_scratch/slime_rl"
    print(f"Results Directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)
    rng, env_rng = jax.random.split(rng)

    env = gym.make(cfg.env.name)
    print(f"Initialized environment: {cfg.env.name}")

    num_inputs = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        num_output = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.MultiBinary) and env.action_space.n == 3:
         print("Treating MultiBinary(3) action space as Discrete(8)")
         num_output = 8
    else:
        raise ValueError(f"Unsupported action space type: {type(env.action_space)}")

    cfg.network.num_inputs = num_inputs
    cfg.network.num_output = num_output
    print(f"Observation Space Size (num_inputs): {num_inputs}")
    print(f"Action Space Size (num_output): {num_output}")

    current_layer_sizes = list(cfg.network.num_layers)
    current_activations = []
    rng, model_rng = jax.random.split(rng)
    _, current_activations = create_layers(model_rng, current_layer_sizes, num_output, None)
    print(f"Initial Hidden Layer Sizes: {current_layer_sizes}")
    print(f"Initial Activations: {[a.__name__ for a in current_activations]}")

    model, params = copy_params_and_create_model(model_rng, current_layer_sizes, current_activations, num_output, num_inputs, source_params=None)
    optimizer = optax.adam(learning_rate=cfg.training.lr)

    reward_summary_file = os.path.join(results_dir, "generation_summary.csv")
    detailed_log_file = os.path.join(results_dir, "detailed_log.json")
    detailed_log_data = []
    generation_avg_rewards = []

    with open(reward_summary_file, 'w', newline='') as f:
         writer = csv.writer(f)
         writer.writerow([
             "generation", "average_reward", "num_hidden_layers",
             "hidden_layer_sizes", "activation_functions"
         ])

    for generation in range(cfg.training.generations):
        print(f"\n--- Starting Generation {generation} ---")
        print(f"Current Structure: Layers={current_layer_sizes}, Activations={[a.__name__ for a in current_activations]}")

        current_params_struct = params if 'params' in params else {'params': params}
        model_state = train_state.TrainState.create(apply_fn=model.apply, params=current_params_struct, tx=optimizer)

        all_obs, all_acts, all_rews, all_log_probs, episode_rewards = [], [], [], [], []
        print(f"Collecting experience for {cfg.training.episodes_per_generation} episodes...")
        start_collect_time = time.time()
        for _ in tqdm(range(cfg.training.episodes_per_generation), desc="Episodes"):
            rng, episode_rng = jax.random.split(rng)
            obs, acts, rews, log_ps, ep_rew, rng = run_episode(env, model_state, episode_rng)
            all_obs.extend(obs)
            all_acts.extend(acts)
            all_rews.append(rews)
            all_log_probs.extend(log_ps)
            episode_rewards.append(ep_rew)
        collect_time = time.time() - start_collect_time
        avg_reward_gen = np.mean(episode_rewards) if episode_rewards else 0.0
        generation_avg_rewards.append(avg_reward_gen)
        print(f"Collection complete. Avg Reward: {avg_reward_gen:.2f} (Time: {collect_time:.2f}s)")

        if not all_obs:
            print("Warning: No experience collected, skipping update and mutations.")
            continue

        observations_np = jnp.array(all_obs)
        actions_np = jnp.array(all_acts)
        log_probs_old_np = jnp.array(all_log_probs)

        all_returns = []
        for rews in all_rews:
            returns = calculate_discounted_returns(rews, cfg.training.gamma)
            all_returns.extend(returns)
        returns_np = jnp.array(all_returns)

        if not (observations_np.shape[0] == actions_np.shape[0] == returns_np.shape[0]):
             print(f"Error: Mismatch in collected data shapes! Obs={observations_np.shape[0]}, Acts={actions_np.shape[0]}, Rets={returns_np.shape[0]}")
             continue

        print("Updating policy...")
        start_update_time = time.time()
        model_state, loss = train_step_rl(model_state, observations_np, actions_np, log_probs_old_np, returns_np)
        update_time = time.time() - start_update_time
        print(f"Update complete. Loss: {loss:.4f} (Time: {update_time:.2f}s)")

        with open(reward_summary_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                f"{avg_reward_gen:.6f}",
                len(current_layer_sizes),
                str(current_layer_sizes),
                str([a.__name__ for a in current_activations])
            ])

        state_dict_params = serialization.to_state_dict(model_state.params)
        serializable_params = convert_params_to_json_serializable(
            state_dict_params.get('params', {}))

        gen_data = {
            "generation": generation,
            "average_reward": float(avg_reward_gen),
            "structure": {
                "input_size": cfg.network.num_inputs,
                "output_size": cfg.network.num_output,
                "hidden_layer_sizes": current_layer_sizes,
                "activation_functions": [act.__name__ for act in current_activations],
                "parameters": serializable_params
            },
            "training_stats": {
                "loss": float(loss),
                "episodes_collected": cfg.training.episodes_per_generation,
                "steps_collected": len(observations_np),
                "collection_time_sec": collect_time,
                "update_time_sec": update_time
            }
        }
        detailed_log_data.append(gen_data)

        if cfg.utils.draw_graph and generation % cfg.utils.draw_graph_freq == 0:
            draw_graph(model_state, current_activations, cfg)
            graph_filename = os.path.join(results_dir, f"graph_gen{generation}.png")
            plt.savefig(graph_filename)
            plt.close()

        # --- Evolution / Mutation ---
        print("\n--- Applying Mutations ---")
        rng, evo_rng = jax.random.split(rng)
        mutation_rngs = jax.random.split(evo_rng, 5)
        current_params_for_mutation = model_state.params['params']

        next_model = model
        next_params = model_state.params
        next_layer_sizes = current_layer_sizes
        next_activations = current_activations
        model_updated = False

        # Apply mutations based on probabilities
        if jax.random.uniform(mutation_rngs[0]) < cfg.neat.add_layer:
            print("Attempting Add Layer...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = add_new_layer(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
            model_updated = True
        if jax.random.uniform(mutation_rngs[1]) < cfg.neat.add_node:
            print("Attempting Add Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = add_new_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
            model_updated = True
        if jax.random.uniform(mutation_rngs[2]) < cfg.neat.remove_node:
            print("Attempting Remove Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
            model_updated = True
        if jax.random.uniform(mutation_rngs[3]) < cfg.neat.remove_layer:
            print("Attempting Remove Layer...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_layer(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
            model_updated = True
        if jax.random.uniform(mutation_rngs[4]) < cfg.neat.mutate_activation:
            print("Attempting Mutate Activation...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = mutate_activation(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
            model_updated = True

        model = next_model
        params = next_params
        current_layer_sizes = next_layer_sizes
        current_activations = next_activations

    print("\n--- Training Complete ---")
    print(f"Reward summary saved to: {reward_summary_file}")
    print(f"Graphs saved in: {results_dir}")

    with open(detailed_log_file, 'w') as f:
        json.dump(detailed_log_data, f, indent=4)
    print(f"Detailed structure and reward log saved to: {detailed_log_file}")

    generations_from_csv = []
    rewards_from_csv = []
    with open(reward_summary_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                generations_from_csv.append(int(row[0]))
                rewards_from_csv.append(float(row[1]))


    if generations_from_csv and rewards_from_csv:
         plt.figure(figsize=(10, 5))
         plt.plot(generations_from_csv, rewards_from_csv, marker='o')
         plt.title("Average Episode Reward over Generations")
         plt.xlabel("Generation")
         plt.ylabel("Average Reward")
         plt.grid(True)
         plot_filename = os.path.join(results_dir, "reward_plot.png")
         plt.savefig(plot_filename)
         plt.close()
         print(f"Reward plot saved to: {plot_filename}")
    else:
         print("No data found in CSV to generate reward plot.")


    env.close()

if __name__ == "__main__":
    main()
