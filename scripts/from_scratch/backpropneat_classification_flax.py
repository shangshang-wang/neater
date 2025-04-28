import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import csv
from functools import partial
import hydra
import flax
from flax import serialization
from flax.training import train_state, checkpoints
from flax import linen as nn
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision.datasets import MNIST
from tqdm import tqdm


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

    if not hasattr(trained_model_state, 'params') or 'params' not in trained_model_state.params:
        print("Warning: Model parameters key ('params') not found in TrainState. Cannot draw graph.")
        plt.figure(figsize=(12, 8))
        plt.title("Graph Generation Skipped (Missing 'params' key)")
        return
    param_dict = trained_model_state.params['params']

    if not isinstance(param_dict, (dict, flax.core.FrozenDict)) or not param_dict:
        print(f"Warning: Parameter dictionary is empty or not a valid dictionary type ({type(param_dict)}). Cannot draw graph.")
        plt.figure(figsize=(12, 8))
        plt.title("Graph Generation Skipped (Empty/Invalid Params Dict)")
        return

    num_elements = cfg.network.num_inputs
    start_nodes = [f'input_node_{i + 1}' for i in range(min(num_elements, 6))]
    if num_elements > 6:
        start_nodes = start_nodes[:3] + ['input_ellipsis'] + start_nodes[-3:]

    for i, node in enumerate(start_nodes):
        if node == 'input_ellipsis':
            y_pos = 0
            G.add_node(node, pos=(0, y_pos), shape='dot')
            G.add_node(f"{node}_dot2", pos=(0, y_pos - 0.12), shape='dot')
            G.add_node(f"{node}_dot3", pos=(0, y_pos + 0.12), shape='dot')
        else:
            if num_elements > 6:
                adj_idx = i if i < 3 else i - 1
                total_vis_nodes = 7
                y_pos = adj_idx - (total_vis_nodes - 1) / 2
            else:
                y_pos = i - (len(start_nodes) - 1) / 2
            G.add_node(node, pos=(0, y_pos))

    prev_layer_node_names_for_edges = [n for n in start_nodes if 'ellipsis' not in n]
    prev_layer_nodes_vis = start_nodes

    layer_distance = 0.5
    max_nodes = 0
    layer_items = list(param_dict.items())

    edge_weights_dict = {}

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
            logical_vis_count = 7 # 3 + ellipsis + 3 for positioning

        current_layer_node_names_for_edges = []
        current_layer_nodes_vis = []

        for node_i_vis in range(logical_vis_count):
            is_ellipsis = use_ellipsis_current and node_i_vis == 3
            node_actual_idx = vis_indices_current[node_i_vis] if not is_ellipsis else -1
            current_node_name = f"{layer}_node_{node_actual_idx}" if not is_ellipsis else f"{layer}_ellipsis"
            y_pos = node_i_vis - (logical_vis_count - 1) / 2
            current_layer_nodes_vis.append(current_node_name)

            if is_ellipsis:
                G.add_node(current_node_name, pos=(layer_distance, y_pos), shape='dot')
                G.add_node(f"{current_node_name}_dot2", pos=(layer_distance, y_pos - 0.12), shape='dot')
                G.add_node(f"{current_node_name}_dot3", pos=(layer_distance, y_pos + 0.12), shape='dot')
            else:
                G.add_node(current_node_name, pos=(layer_distance, y_pos))
                current_layer_node_names_for_edges.append(current_node_name)

                for prev_node_name in prev_layer_node_names_for_edges:
                     parts = prev_node_name.split('_')
                     if parts[-2] == 'node':
                         prev_node_actual_idx = int(parts[-1])
                     else:
                         continue

                     if 0 <= prev_node_actual_idx < num_prev_nodes_actual and 0 <= node_actual_idx < num_nodes_actual:
                         weight_value = value['kernel'][prev_node_actual_idx][node_actual_idx]
                         G.add_edge(prev_node_name, current_node_name, weight=weight_value)
                         edge_weights_dict[(prev_node_name, current_node_name)] = f"{weight_value:.2f}"

        prev_layer_node_names_for_edges = current_layer_node_names_for_edges
        prev_layer_nodes_vis = current_layer_nodes_vis
        layer_distance += 0.5

    plt.figure(figsize=(15, 10))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
         print("Warning: Graph position attribute is empty. Skipping draw.")
         plt.title("Graph Generation Skipped (Empty Positions)")
         plt.close()
         return

    distance = 0.5
    count = 1
    max_y = 0
    min_y = 0
    if pos:
        y_coords = [p[1] for p in pos.values()]
        if y_coords:
            max_y = max(y_coords)
            min_y = min(y_coords)

    height_text = max_y + 2.0 if y_coords else 2.0
    activation_text_height = height_text - 0.5

    layer_items_for_text = list(param_dict.items())

    for i, (layer, value) in enumerate(layer_items_for_text):
         if 'kernel' not in value: continue
         num_nodes_actual = value['kernel'].shape[1]
         current_layer_x = 0.5 * (i + 1)

         if i < len(layer_items_for_text) - 1:
            plt.text(current_layer_x, height_text, f"HL {count}", fontsize=11, color='black', ha='center', va='center')
            plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')
            if i < len(activations_list):
                activation_name = activations_list[i].__name__
                plt.text(current_layer_x, activation_text_height, f"act: {activation_name}", fontsize=9, color='blue', ha='center', va='center')
            count += 1
         else:
             plt.text(current_layer_x, height_text, f"Output Layer", fontsize=11, color='black', ha='center', va='center')
             plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')

    input_layer_x = 0
    plt.text(input_layer_x, height_text, f"Input Layer", fontsize=11, color='black', ha='center', va='center')
    plt.text(input_layer_x, height_text - 0.25, f"nodes: {cfg.network.num_inputs}", fontsize=9, color='black', ha='center', va='center')

    edge_weights_vals = [abs(d.get('weight', 0)) for u, v, d in G.edges(data=True)]
    max_edge_width = max(edge_weights_vals) if edge_weights_vals else 1
    edge_widths_normalized = [(abs(w) / max_edge_width * 2.5 + 0.2) if max_edge_width > 0 else 0.2 for w in edge_weights_vals]

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6',
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') != 'dot'])
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='grey',
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') == 'dot'])
    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized, alpha=0.5, node_size=400)

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_weights_dict,
        font_size=7,
        font_color='red',
        label_pos=0.1,
        bbox=dict(alpha=0)
    )

    plt.tight_layout()
    plt.axis('off')


def calculate_loss_acc(state, params, batch, num_output):
    data_input, labels = batch
    model_params = params['params'] if 'params' in params else params

    logits = state.apply_fn({'params': model_params}, data_input)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    probs = jax.nn.softmax(logits)
    max_index = jnp.argmax(probs, axis=-1)

    pred_labels = jax.nn.one_hot(max_index, num_output)
    acc = jnp.all(pred_labels == labels, axis=-1).mean()

    return loss, acc

@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, num_output):
    grad_fn = jax.value_and_grad(calculate_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch, num_output)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

def train_model(state, train_data_loader, test_data_loader, num_epochs, generation, num_output):
    best_epoch_acc = -1.0
    for epoch in tqdm(range(num_epochs), desc=f"Gen {generation} Training"):
        batch_losses = []
        batch_accs = []
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch, num_output)
            batch_losses.append(loss)
            batch_accs.append(acc)

        epoch_test_acc = eval_model(state, test_data_loader, num_output)
        best_epoch_acc = max(best_epoch_acc, epoch_test_acc)

    final_test_acc = eval_model(state, test_data_loader, num_output)
    print(f"Generation {generation} Final Test Accuracy: {final_test_acc:.4f} (Best during epoch: {best_epoch_acc:.4f})")

    return state, final_test_acc

@partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, num_output):
    _, acc = calculate_loss_acc(state, state.params, batch, num_output)
    return acc

def eval_model(state, data_loader, num_output):
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch, num_output)
        all_accs.append(batch_acc)
        input_data = batch[0]
        if isinstance(input_data, torch.Tensor):
             input_data = input_data.numpy()
        batch_sizes.append(input_data.shape[0])

    acc = sum([a * b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    return float(acc)

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class IrisDataset(Dataset):
    def __init__(self, X, y, num_classes=3):
        self.X = X
        self.y = jax.nn.one_hot(y, num_classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DigitsDataset(Dataset):
    def __init__(self, X, y, num_classes=10):
        self.X = X
        self.y = jax.nn.one_hot(y, num_classes)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def mnist_transform(x):
    np_img = np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.
    return np_img.flatten()

def mnist_collate_fn(batch):
    batch = list(zip(*batch))
    x = np.stack(batch[0])
    y = jax.nn.one_hot(np.array(batch[1]), 10)
    return x, y


POSSIBLE_ACTIVATIONS = {
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'leaky_relu': nn.leaky_relu,
    'swish': nn.swish,
    'gelu': nn.gelu
}
POSSIBLE_ACTIVATION_FNS = list(POSSIBLE_ACTIVATIONS.values())
POSSIBLE_ACTIVATION_NAMES = list(POSSIBLE_ACTIVATIONS.keys())

def copy_params_and_create_model(rng, layer_sizes, current_activations, num_output, num_inputs, source_params=None):
    rng, layer_rng, init_rng, input_rng = jax.random.split(rng, 4)

    layers, activations = create_layers(layer_rng, layer_sizes, num_output, current_activations)
    new_model = GenomeClassifier(layer_definitions=layers, activation_fns=activations)

    dummy_input = jax.random.normal(input_rng, (num_inputs,))
    target_params = new_model.init(init_rng, dummy_input)['params']
    new_params = target_params

    if source_params is None:
        print("No source parameters provided, returning newly initialized model.")
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

        print(f"Copying layer {i}: {source_layer_name} -> {target_layer_name}")

        if 'kernel' in source_layer_values and 'kernel' in target_layer_values:
            source_kernel = source_layer_values['kernel']
            target_kernel = target_layer_values['kernel']

            # assert isinstance(source_kernel, jnp.ndarray), f"Expected JAX array, got {type(source_kernel)}"

            copy_rows = min(source_kernel.shape[0], target_kernel.shape[0])
            copy_cols = min(source_kernel.shape[1], target_kernel.shape[1])

            updated_kernel = target_kernel.copy()
            updated_kernel = updated_kernel.at[:copy_rows, :copy_cols].set(source_kernel[:copy_rows, :copy_cols])

            new_params_mutable[target_layer_name]['kernel'] = updated_kernel

        if 'bias' in source_layer_values and 'bias' in target_layer_values:
            source_bias = source_layer_values['bias']
            target_bias = target_layer_values['bias']

            copy_len = min(len(source_bias), len(target_bias))

            updated_bias = target_bias.copy()
            updated_bias = updated_bias.at[:copy_len].set(source_bias[:copy_len])
            new_params_mutable[target_layer_name]['bias'] = updated_bias

    return new_model, {'params': flax.core.freeze(new_params_mutable)}

def create_layers(rng, layer_sizes, num_output, current_activations=None):
    layers = []
    activations = []
    num_hidden_layers = len(layer_sizes)

    if current_activations is not None and len(current_activations) != num_hidden_layers:
        print(f"Warning: Provided activation list length ({len(current_activations)}) "
              f"doesn't match number of hidden layers ({num_hidden_layers}). Re-initializing.")
        current_activations = None

    for i, hidden_size in enumerate(layer_sizes):
        layers.append(nn.Dense(features=hidden_size))
        if current_activations is None:
            rng, act_rng = jax.random.split(rng)
            num_possible_activations = len(POSSIBLE_ACTIVATION_FNS)
            indices_array = jnp.arange(num_possible_activations)
            chosen_index = jax.random.choice(act_rng, indices_array)
            chosen_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
            activations.append(chosen_activation)
        else:
            activations.append(current_activations[i])

    layers.append(nn.Dense(features=num_output))
    return layers, activations

def add_new_layer(rng, layer_sizes, current_activations, source_params, cfg):
    rng, insert_rng, act_rng = jax.random.split(rng, 3)
    insert_pos = jax.random.randint(insert_rng, (1,), 0, len(layer_sizes) + 1).item()
    new_layer_size = 1
    new_layer_sizes = layer_sizes[:insert_pos] + [new_layer_size] + layer_sizes[insert_pos:]

    num_possible_activations = len(POSSIBLE_ACTIVATION_FNS)
    indices_array = jnp.arange(num_possible_activations)
    chosen_index = jax.random.choice(act_rng, indices_array)

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
        print("  Skipping Remove Node: No eligible layers found (all have size 1 or fewer).")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations

    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.choice(choice_rng, jnp.array(eligible_layers_indices)).item()

    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] -= 1
    new_activations = current_activations
    print(f"  Decremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")

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

    return copy_params_and_create_model(rng, layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), layer_sizes, new_activations


class GenomeClassifier(nn.Module):
    layer_definitions: list
    activation_fns: list

    @nn.compact
    def __call__(self, x):
        if len(self.layer_definitions) != len(self.activation_fns) + 1:
             raise ValueError(f"Mismatch between layers ({len(self.layer_definitions)}) and activations ({len(self.activation_fns)})")

        for i in range(len(self.activation_fns)):
            x = self.layer_definitions[i](x)
            x = self.activation_fns[i](x)

        x = self.layer_definitions[-1](x)
        return x

class SimpleClassifier(nn.Module):
    num_hidden: int
    num_outputs: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


# @hydra.main(version_base=None, config_path="../../task", config_name="config_iris")
@hydra.main(version_base=None, config_path="../../task", config_name="config_mnist")
def main(cfg):
    results_dir = f"./assets/results/from_scratch/{cfg.dataset.dataset_type}"
    print(f"Results dir: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)
    current_layer_sizes = list(cfg.network.num_layers)
    current_activations = []
    num_inputs = cfg.network.num_inputs
    num_output = cfg.network.num_output
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.lr

    rng, model_rng = jax.random.split(rng)
    _, current_activations = create_layers(model_rng, current_layer_sizes, num_output, None)
    print(f"Initial Hidden Layer Sizes: {current_layer_sizes}")
    print(f"Initial Activations: {[a.__name__ for a in current_activations]}")
    model, params = copy_params_and_create_model(
        model_rng, current_layer_sizes, current_activations, num_output, num_inputs, source_params=None
    )
    optimizer = optax.sgd(learning_rate=learning_rate)

    if cfg.dataset.dataset_type == "iris":
        print("Loading Iris Dataset")
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_std = np.where(X_std == 0, 1, X_std)
        X = (X - X_mean) / X_std
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=cfg.jax.PRNGKey, stratify=y) # Stratify for iris
        train_dataset = IrisDataset(X_train, y_train, num_classes=num_output)
        test_dataset = IrisDataset(X_test, y_test, num_classes=num_output)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False)
    elif cfg.dataset.dataset_type == "mnist":
        print("Loading MNIST Dataset")
        train = MNIST(root='train', train=True, transform=mnist_transform, download=True)
        test = MNIST(root='test', train=False, transform=mnist_transform, download=True)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=mnist_collate_fn, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=mnist_collate_fn, drop_last=False)
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_type}")

    accuracy_file = os.path.join(results_dir, "generation_summary.csv")
    detailed_log_file = os.path.join(results_dir, "detailed_log.json")
    detailed_log_data = []

    for generation in range(cfg.training.generations):
        print(f"\n--- Starting Generation {generation} ---")
        print(f"Current Structure: Layers={current_layer_sizes}, Activations={[a.__name__ for a in current_activations]}")

        current_params = params if 'params' in params else {'params': params}
        model_state = train_state.TrainState.create(apply_fn=model.apply, params=current_params, tx=optimizer)

        trained_model_state, gen_test_accuracy = train_model(
            state=model_state,
            train_data_loader=train_loader,
            test_data_loader=test_loader,
            num_epochs=cfg.training.num_epochs,
            generation=generation,
            num_output=num_output
        )

        with open(accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                f"{gen_test_accuracy:.6f}",
                len(current_layer_sizes),
                str(current_layer_sizes),
                str([a.__name__ for a in current_activations])
            ])

        state_dict_params = serialization.to_state_dict(trained_model_state.params)
        serializable_params = convert_params_to_json_serializable(
            state_dict_params.get('params', {}))

        gen_data = {
            "generation": generation,
            "accuracy": float(gen_test_accuracy),
            "structure": {
                "input_size": cfg.network.num_inputs,
                "output_size": cfg.network.num_output,
                "hidden_layer_sizes": current_layer_sizes,
                "activation_functions": [act.__name__ for act in current_activations],
                "parameters": serializable_params
            }
        }
        detailed_log_data.append(gen_data)

        if cfg.utils.draw_graph:
            draw_graph(trained_model_state, current_activations, cfg)
            graph_filename = os.path.join(results_dir, f"nextwork-generation-{generation}.png")
            plt.savefig(graph_filename)
            plt.close()
            print(f"Saved graph for generation {generation} to {graph_filename}")

        print("\n--- Applying Mutations ---")
        rng, evo_rng = jax.random.split(rng)
        mutation_rngs = jax.random.split(evo_rng, 5)
        current_params_for_mutation = trained_model_state.params
        if 'params' in current_params_for_mutation:
            current_params_for_mutation = current_params_for_mutation['params']

        next_model = model
        next_params = trained_model_state.params
        next_layer_sizes = current_layer_sizes
        next_activations = current_activations

        # Add new layer
        if jax.random.uniform(mutation_rngs[0]) < cfg.neat.add_layer:
            print("Attempting Add Layer...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = add_new_layer(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg)

        # Add new node
        if jax.random.uniform(mutation_rngs[1]) < cfg.neat.add_node:
            print("Attempting Add Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = add_new_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg)

        # Remove node
        if jax.random.uniform(mutation_rngs[2]) < cfg.neat.remove_node:
            print("Attempting Remove Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg)

        # Remove layer
        if jax.random.uniform(mutation_rngs[3]) < cfg.neat.remove_layer:
            print("Attempting Remove Layer...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_layer(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg)

        if jax.random.uniform(mutation_rngs[4]) < cfg.neat.mutate_activation:
            print("Attempting Mutate Activation...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = mutate_activation(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg)

        model = next_model
        params = next_params
        current_layer_sizes = next_layer_sizes
        current_activations = next_activations

    print("\n--- Training Complete ---")
    print(f"Simple summary saved to: {accuracy_file}")
    print(f"Graphs saved in: {results_dir}")

    with open(detailed_log_file, 'w') as f:
        json.dump(detailed_log_data, f, indent=4)
    print(f"Detailed structure and accuracy log saved to: {detailed_log_file}")

    generations_from_csv = []
    accuracies_from_csv = []
    with open(accuracy_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            generations_from_csv.append(int(row[0]))
            accuracies_from_csv.append(float(row[1]))

    plt.figure(figsize=(10, 5))
    plt.plot(generations_from_csv, accuracies_from_csv, marker='o')
    plt.title("Test Accuracy over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))
    plt.close()
    print(f"Accuracy plot saved to: {os.path.join(results_dir, 'accuracy_plot.png')}")


if __name__ == "__main__":
    main()
