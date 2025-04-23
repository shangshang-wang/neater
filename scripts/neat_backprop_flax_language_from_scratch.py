# Single file implementation for NEAT+Backprop on a simple Language Task
# Based on previous multi-task script, streamlined for next-word prediction.
# Incorporates fixes for parameter dictionary structure (KeyError: 'params').
# Removed semicolons as requested.

import hydra
from omegaconf import DictConfig, OmegaConf # For Hydra config management

import os
import csv
import json
import pickle
import re
from collections import Counter
from functools import partial
import flax
from flax.training import train_state, checkpoints
from flax import linen as nn
from flax import serialization # For converting params to dict
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import optax
from sklearn.model_selection import train_test_split
import torch # Still needed for DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Constants ---
POSSIBLE_ACTIVATIONS = {
    'relu': nn.relu,
    'sigmoid': nn.sigmoid,
    'tanh': nn.tanh,
    'leaky_relu': nn.leaky_relu,
    'swish': nn.swish, # Also known as silu
    'gelu': nn.gelu
}
POSSIBLE_ACTIVATION_FNS = list(POSSIBLE_ACTIVATIONS.values())
POSSIBLE_ACTIVATION_NAMES = list(POSSIBLE_ACTIVATIONS.keys())
WEIGHT_LABEL_THRESHOLD = 0.5 # For filtering edge labels in visualization

# --- Language Data Preprocessing ---

def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    # Keep letters, numbers, spaces, periods. Remove other punctuation.
    text = re.sub(r'[^a-z0-9\s\.]', '', text)
    # Ensure space around periods for tokenization
    text = re.sub(r'\.', r' . ', text)
    text = re.sub(r' +', ' ', text).strip() # Remove extra spaces
    return text

def build_vocab(text, min_freq=1):
    """Builds vocabulary mapping words to indices."""
    words = text.split(' ')
    # Filter out empty strings that might result from splitting
    words = [word for word in words if word]
    word_counts = Counter(words)
    # Start vocab with PAD and UNK tokens
    vocab = {
        '<PAD>': 0, # Padding token
        '<UNK>': 1  # Unknown token
    }
    idx = 2
    # Sort words by frequency for potential future use (optional)
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    for word, count in sorted_words:
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    print(f"Vocabulary Size: {len(vocab)}")
    return vocab, {v: k for k, v in vocab.items()} # Return vocab and inverse vocab


def create_sequences(text, vocab, context_window):
    """Creates context window sequences and next-word targets."""
    words = text.split(' ')
    words = [word for word in words if word] # Ensure no empty strings
    word_indices = [vocab.get(word, vocab['<UNK>']) for word in words]

    sequences = []
    targets = []
    if len(word_indices) <= context_window:
        print(f"Warning: Text length ({len(word_indices)}) <= context window ({context_window}). No sequences generated.")
        return np.array([]), np.array([])

    for i in range(len(word_indices) - context_window):
        context = word_indices[i : i + context_window]
        target = word_indices[i + context_window]
        sequences.append(context)
        targets.append(target)

    if not sequences: # Handle case where loop doesn't run
         return np.array([]), np.array([])

    return np.array(sequences, dtype=np.int32), np.array(targets, dtype=np.int32)


class LanguageDataset(Dataset):
    """PyTorch Dataset for the language task."""
    def __init__(self, sequences, targets, vocab_size):
        self.sequences = sequences # Shape: (num_seq, context_window)
        self.targets = targets     # Shape: (num_seq,)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Targets should be one-hot encoded for softmax cross-entropy
        one_hot_target = jax.nn.one_hot(self.targets[idx], num_classes=self.vocab_size)
        # Return context indices (sequence) and one-hot target
        return self.sequences[idx], one_hot_target


# --- JAX/Flax Model Definitions ---

class GenomeClassifier(nn.Module):
    """The evolvable feed-forward part of the network (Dense layers)."""
    layer_definitions: list # List of layer objects (e.g., nn.Dense instances)
    activation_fns: list    # List of activation functions (callable)

    @nn.compact
    def __call__(self, x):
        # Expects flattened input: (batch_size, context_window * embedding_dim)
        if not self.layer_definitions: # Handle case with no hidden layers
             raise ValueError("GenomeClassifier requires at least an output layer definition.")

        if len(self.layer_definitions) != len(self.activation_fns) + 1:
             raise ValueError(f"Mismatch: {len(self.layer_definitions)} layers, {len(self.activation_fns)} activations")

        # Apply hidden layers and activations
        for i in range(len(self.activation_fns)):
            x = self.layer_definitions[i](x)
            x = self.activation_fns[i](x) # Apply corresponding activation

        # Apply final output layer (no activation here, softmax is in loss)
        x = self.layer_definitions[-1](x)
        return x


class LanguageModelWrapper(nn.Module):
    """Wraps the embedding layer and the evolvable GenomeClassifier."""
    genome_classifier: nn.Module # The evolved Dense part instance
    vocab_size: int
    embedding_dim: int
    # context_window is implicit in the flattened input size of genome_classifier

    def setup(self):
        # Embedding layer
        self.embed = nn.Embed(num_embeddings=self.vocab_size,
                              features=self.embedding_dim)

    def __call__(self, x_indices):
        # x_indices shape: (batch_size, context_window)
        assert x_indices.ndim == 2, f"Input should have shape (batch, context_window), got {x_indices.shape}"
        assert x_indices.dtype == jnp.int32 or x_indices.dtype == jnp.int64, f"Input indices should be integers, got {x_indices.dtype}"

        embedded_x = self.embed(x_indices) # shape: (batch_size, context_window, embedding_dim)

        # Flatten the context window embeddings
        batch_size = embedded_x.shape[0]
        # Calculate expected flattened size based on genome_classifier input
        # This assumes genome_classifier expects context_window * embedding_dim
        flattened_x = embedded_x.reshape((batch_size, -1))

        # Pass flattened input to the evolved classifier
        logits = self.genome_classifier(flattened_x) # shape: (batch_size, vocab_size)
        return logits


# --- NEAT / Evolution Functions ---

def create_layers(rng, layer_sizes, num_output, current_activations=None):
    """Creates Dense layers and activation functions for GenomeClassifier."""
    layers = []
    activations = []
    num_hidden_layers = len(layer_sizes)

    # Ensure current_activations has the correct length if provided
    if current_activations is not None and len(current_activations) != num_hidden_layers:
        print(f"Warning: Provided activation list length ({len(current_activations)}) "
              f"doesn't match number of hidden layers ({num_hidden_layers}). Re-initializing.")
        current_activations = None # Force re-initialization

    for i, hidden_size in enumerate(layer_sizes):
        layers.append(nn.Dense(features=hidden_size))
        if current_activations is None:
            # Choose a random activation function index
            rng, act_rng = jax.random.split(rng)
            num_possible_activations = len(POSSIBLE_ACTIVATION_FNS)
            indices_array = jnp.arange(num_possible_activations)
            chosen_index = jax.random.choice(act_rng, indices_array)
            chosen_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
            activations.append(chosen_activation)
        else:
            activations.append(current_activations[i])

    # Output layer (for GenomeClassifier)
    layers.append(nn.Dense(features=num_output))

    return layers, activations

def copy_params_and_create_model(rng, layer_sizes, current_activations, num_output, num_inputs, source_gc_params=None):
    """
    Creates a new GenomeClassifier instance and copies parameters from a source GC.
    Parameters: source_gc_params (dict or None): Structure: {'layer_definitions_0': ...}
    Returns: tuple: (new_genome_classifier_instance, new_gc_params_dict) where new_gc_params_dict is {'layer_definitions_0': ...}
    """
    rng, layer_rng, init_rng, input_rng = jax.random.split(rng, 4)
    gc_layers, gc_activations = create_layers(layer_rng, layer_sizes, num_output, current_activations)
    new_gc_model = GenomeClassifier(layer_definitions=gc_layers, activation_fns=gc_activations)
    dummy_gc_input = jax.random.normal(input_rng, (num_inputs,))
    try:
        target_gc_params = new_gc_model.init(init_rng, dummy_gc_input)['params']
    except Exception as e:
         print(f"Error during GC initialization:\n  Sizes: {layer_sizes}, Activations: {[a.__name__ for a in gc_activations if hasattr(a, '__name__')]}\n  Input Shape: {dummy_gc_input.shape}\n  Error: {e}")
         raise e

    new_gc_params = target_gc_params
    if source_gc_params is None:
        return new_gc_model, new_gc_params # Return unwrapped params

    # --- Parameter Copying Logic ---
    new_gc_params_mutable = flax.core.unfreeze(new_gc_params)

    if isinstance(source_gc_params, flax.core.FrozenDict):
        source_gc_params_dict = source_gc_params.unfreeze()
    elif isinstance(source_gc_params, dict):
        source_gc_params_dict = source_gc_params
    else:
        print(f"Warning: Unexpected source_gc_params type: {type(source_gc_params)}.")
        source_gc_params_dict = {}

    source_params_layers = list(source_gc_params_dict.items())
    target_params_layers = list(new_gc_params_mutable.items())
    num_layers_to_copy = min(len(source_params_layers), len(target_params_layers))

    for i in range(num_layers_to_copy):
        source_layer_name, source_layer_values = source_params_layers[i]
        target_layer_name, target_layer_values = target_params_layers[i]

        # Check if source/target layers are actually parameter dicts
        if not isinstance(source_layer_values, (dict, flax.core.FrozenDict)) or \
           not isinstance(target_layer_values, (dict, flax.core.FrozenDict)):
             print(f"Warning: Skipping copy for layer {i}, unexpected param structure: {type(source_layer_values)}, {type(target_layer_values)}")
             continue

        if 'kernel' in source_layer_values and 'kernel' in target_layer_values:
            source_kernel = source_layer_values['kernel']
            target_kernel = target_layer_values['kernel']
            # Basic shape check
            if not hasattr(source_kernel, 'shape') or not hasattr(target_kernel, 'shape'):
                continue
            copy_rows = min(source_kernel.shape[0], target_kernel.shape[0])
            copy_cols = min(source_kernel.shape[1], target_kernel.shape[1])
            updated_kernel = target_kernel.copy()
            # Ensure slice indices are non-negative
            if copy_rows > 0 and copy_cols > 0:
                updated_kernel = updated_kernel.at[:copy_rows, :copy_cols].set(source_kernel[:copy_rows, :copy_cols])
            new_gc_params_mutable[target_layer_name]['kernel'] = updated_kernel

        if 'bias' in source_layer_values and 'bias' in target_layer_values:
            source_bias = source_layer_values['bias']
            target_bias = target_layer_values['bias']
            if not hasattr(source_bias, 'shape') or not hasattr(target_bias, 'shape'):
                continue
            copy_len = min(len(source_bias), len(target_bias))
            updated_bias = target_bias.copy()
            if copy_len > 0:
                updated_bias = updated_bias.at[:copy_len].set(source_bias[:copy_len])
            new_gc_params_mutable[target_layer_name]['bias'] = updated_bias

    return new_gc_model, flax.core.freeze(new_gc_params_mutable) # Return unwrapped params


# --- Mutation Functions (Operate on GC structure and params) ---
# They now receive and return the unwrapped GC param dict {'layer_definitions_0': ...}

def add_new_layer(rng, layer_sizes, current_activations, source_gc_params, cfg):
    """ Adds a new hidden layer to GenomeClassifier. """
    rng, insert_rng, act_rng = jax.random.split(rng, 3)
    insert_pos = jax.random.randint(insert_rng, (1,), 0, len(layer_sizes) + 1).item()
    new_layer_size = 1
    new_layer_sizes = layer_sizes[:insert_pos] + [new_layer_size] + layer_sizes[insert_pos:]
    indices_array = jnp.arange(len(POSSIBLE_ACTIVATION_FNS))
    chosen_index = jax.random.choice(act_rng, indices_array)
    new_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
    new_activations = current_activations[:insert_pos] + [new_activation] + current_activations[insert_pos:]
    print(f"  GC New Layer Sizes: {new_layer_sizes}")
    print(f"  GC New Activations: {[a.__name__ for a in new_activations]}")
    gc_num_inputs = cfg.network.num_inputs
    gc_num_output = cfg.network.num_output
    # copy_params_and_create_model returns (gc_instance, gc_params_dict)
    (new_gc_instance, new_gc_params_dict) = copy_params_and_create_model(rng, new_layer_sizes, new_activations, gc_num_output, gc_num_inputs, source_gc_params)
    return (new_gc_instance, new_gc_params_dict), new_layer_sizes, new_activations

def add_new_node(rng, layer_sizes, current_activations, source_gc_params, cfg):
    """ Adds a new node to a random GC hidden layer. """
    gc_num_inputs = cfg.network.num_inputs
    gc_num_output = cfg.network.num_output
    if not layer_sizes:
        print("  Skipping Add Node (GC): No hidden layers exist.")
        (new_model, new_params) = copy_params_and_create_model(rng, layer_sizes, current_activations, gc_num_output, gc_num_inputs, source_gc_params)
        return (new_model, new_params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.randint(choice_rng, (1,), 0, len(layer_sizes)).item()
    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] += 1
    print(f"  GC Incremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")
    new_activations = current_activations # Activations don't change
    (new_gc_instance, new_gc_params_dict) = copy_params_and_create_model(rng, new_layer_sizes, new_activations, gc_num_output, gc_num_inputs, source_gc_params)
    return (new_gc_instance, new_gc_params_dict), new_layer_sizes, new_activations

def remove_node(rng, layer_sizes, current_activations, source_gc_params, cfg):
    """ Removes a node from a random eligible GC hidden layer. """
    gc_num_inputs = cfg.network.num_inputs
    gc_num_output = cfg.network.num_output
    eligible_layers_indices = [i for i, size in enumerate(layer_sizes) if size > 1]
    if not eligible_layers_indices:
        print("  Skipping Remove Node (GC): No eligible layers found.")
        (new_model, new_params) = copy_params_and_create_model(rng, layer_sizes, current_activations, gc_num_output, gc_num_inputs, source_gc_params)
        return (new_model, new_params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.choice(choice_rng, jnp.array(eligible_layers_indices)).item()
    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] -= 1
    print(f"  GC Decremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")
    new_activations = current_activations
    (new_gc_instance, new_gc_params_dict) = copy_params_and_create_model(rng, new_layer_sizes, new_activations, gc_num_output, gc_num_inputs, source_gc_params)
    return (new_gc_instance, new_gc_params_dict), new_layer_sizes, new_activations

def remove_layer(rng, layer_sizes, current_activations, source_gc_params, cfg):
    """ Removes a random GC hidden layer. """
    gc_num_inputs = cfg.network.num_inputs
    gc_num_output = cfg.network.num_output
    if not layer_sizes:
        print("  Skipping Remove Layer (GC): No hidden layers to remove.")
        (new_model, new_params) = copy_params_and_create_model(rng, layer_sizes, current_activations, gc_num_output, gc_num_inputs, source_gc_params)
        return (new_model, new_params), layer_sizes, current_activations
    rng, choice_rng = jax.random.split(rng)
    layer_to_remove_idx = jax.random.randint(choice_rng, (1,), 0, len(layer_sizes)).item()
    new_layer_sizes = layer_sizes[:layer_to_remove_idx] + layer_sizes[layer_to_remove_idx+1:]
    new_activations = current_activations[:layer_to_remove_idx] + current_activations[layer_to_remove_idx+1:]
    print(f"  GC Removed layer idx {layer_to_remove_idx}. New Sizes: {new_layer_sizes}")
    print(f"  GC New Activations: {[a.__name__ for a in new_activations]}")
    (new_gc_instance, new_gc_params_dict) = copy_params_and_create_model(rng, new_layer_sizes, new_activations, gc_num_output, gc_num_inputs, source_gc_params)
    return (new_gc_instance, new_gc_params_dict), new_layer_sizes, new_activations

def mutate_activation(rng, layer_sizes, current_activations, source_gc_params, cfg):
    """ Mutates activation of a random GC hidden layer. """
    gc_num_inputs = cfg.network.num_inputs
    gc_num_output = cfg.network.num_output
    if not current_activations:
        print("  Skipping Mutate Activation (GC): No hidden layers/activations exist.")
        (new_model, new_params) = copy_params_and_create_model(rng, layer_sizes, current_activations, gc_num_output, gc_num_inputs, source_gc_params)
        return (new_model, new_params), layer_sizes, current_activations
    # rng, choice_rng, act_rng = jax.random.split(rng)
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
    print(f"  GC Mutated act layer {layer_to_mutate_idx}: {current_activation_fn.__name__} -> {new_activation_fn.__name__}")
    print(f"  GC New Acts: {[a.__name__ for a in new_activations]}")
    (new_gc_instance, new_gc_params_dict) = copy_params_and_create_model(rng, layer_sizes, new_activations, gc_num_output, gc_num_inputs, source_gc_params)
    return (new_gc_instance, new_gc_params_dict), layer_sizes, new_activations

# --- Training & Evaluation Functions ---

@partial(jax.jit, static_argnums=(3,)) # Jit, static num_output
def calculate_loss_acc(state, params, batch, num_output):
    """Calculates loss and accuracy for language task."""
    # data_input are context indices, labels are one-hot targets
    data_input, labels_one_hot = batch
    try:
        # state.apply_fn points to LanguageModelWrapper.apply
        logits = state.apply_fn(params, data_input) # Pass params directly
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot).mean()
        # Accuracy: fraction of times the highest logit matches the true next word
        pred_index = jnp.argmax(logits, axis=-1)
        true_index = jnp.argmax(labels_one_hot, axis=-1)
        acc = jnp.mean(pred_index == true_index)
    except Exception as e:
         print(f"Error in calculate_loss_acc: {e}")
         return jnp.inf, 0.0
    return loss, acc

@partial(jax.jit, static_argnums=(2,)) # Jit, static num_output
def train_step(state, batch, num_output):
    """Performs a single training step."""
    grad_fn = jax.value_and_grad(calculate_loss_acc, # Function to calculate the loss
                                 argnums=1,  # Differentiate respect to 'params' (second arg)
                                 has_aux=True  # Function returns (loss, acc)
                                 )
    # Pass state (containing apply_fn) and state.params separately
    (loss, acc), grads = grad_fn(state, state.params, batch, num_output)
    state = state.apply_gradients(grads=grads)
    return state, loss, acc

@partial(jax.jit, static_argnums=(2,)) # Jit, static num_output
def eval_step(state, batch, num_output):
    """Performs a single evaluation step."""
    _, acc = calculate_loss_acc(state, state.params, batch, num_output)
    return acc

def eval_model(state, data_loader, num_output):
    """Evaluates the model on the entire dataset."""
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch, num_output)
        all_accs.append(batch_acc)
        input_data = batch[0] # get input data to find batch size
        batch_sizes.append(input_data.shape[0])
    if not batch_sizes or sum(batch_sizes) == 0:
        return 0.0
    acc = sum([a * b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    return float(acc)

def train_model(state, train_data_loader, test_data_loader, num_epochs, generation, num_output):
    """Trains the model for num_epochs."""
    best_epoch_acc = -1.0
    for epoch in tqdm(range(num_epochs), desc=f"Gen {generation} Training", leave=False):
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch, num_output) # Ignore step loss/acc
        # Evaluate on test set periodically
        epoch_test_acc = eval_model(state, test_data_loader, num_output)
        best_epoch_acc = max(best_epoch_acc, epoch_test_acc)
    final_test_acc = eval_model(state, test_data_loader, num_output) # Final eval
    print(f"  Gen {generation} Final Test Acc: {final_test_acc:.4f} (Best: {best_epoch_acc:.4f})")
    return state, final_test_acc # Return state and the final accuracy


# --- Visualization Function ---

def draw_graph(gc_params_dict, activations_list, cfg):
    """ Draws the GenomeClassifier structure. gc_params_dict = {'layer_definitions_0': ...} """
    G = nx.DiGraph()
    param_dict = gc_params_dict # Use the passed GC params dict directly

    if not isinstance(param_dict, (dict, flax.core.FrozenDict)) or not param_dict:
        print(f"Warning: GC Parameter dict invalid in draw_graph.")
        plt.figure(figsize=(12, 8))
        plt.title("Graph Skipped (Invalid GC Params in draw_graph)")
        return

    # --- Node and Edge Creation ---
    num_gc_inputs = cfg.network.num_inputs # Input size for GenomeClassifier
    num_elements = num_gc_inputs
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
            # Simplified y-position calculation
            num_vis_start = len(start_nodes)
            y_pos = i - (num_vis_start -1) / 2 # Center visualized input nodes
            if num_elements > 6 and i >= 4: # Adjust index for lower nodes if ellipsis present
                y_pos = (i-1) - (num_vis_start -1) / 2
            G.add_node(node, pos=(0, y_pos))


    prev_layer_node_names_for_edges = [n for n in start_nodes if 'ellipsis' not in n]
    layer_distance = 0.5
    layer_items = list(param_dict.items()) # Keys are 'layer_definitions_X'
    edge_weights_dict = {}

    for layer_index, (layer, value) in enumerate(layer_items):
        if 'kernel' not in value: continue
        kernel_shape = value['kernel'].shape
        if len(kernel_shape) != 2: continue

        num_prev_nodes_actual, num_nodes_actual = kernel_shape[0], kernel_shape[1]
        num_nodes_vis = min(num_nodes_actual, 6)
        use_ellipsis_current = num_nodes_actual > 6

        # Determine indices and count for visualization
        if use_ellipsis_current:
            vis_indices_current = list(range(3)) + [-1] + list(range(num_nodes_actual - 3, num_nodes_actual))
            logical_vis_count = 7
        else:
            vis_indices_current = list(range(num_nodes_vis))
            logical_vis_count = num_nodes_actual # Use actual count if small

        current_layer_node_names_for_edges = []

        for node_i_vis, node_actual_idx in enumerate(vis_indices_current):
            is_ellipsis = (node_actual_idx == -1)
            current_node_name = f"{layer}_node_{node_actual_idx}" if not is_ellipsis else f"{layer}_ellipsis"
            # Use logical_vis_count for centering y-positions
            y_pos = node_i_vis - (len(vis_indices_current) - 1) / 2 # Position based on # visualized items


            if is_ellipsis:
                G.add_node(current_node_name, pos=(layer_distance, y_pos), shape='dot')
                G.add_node(f"{current_node_name}_dot2", pos=(layer_distance, y_pos - 0.12), shape='dot')
                G.add_node(f"{current_node_name}_dot3", pos=(layer_distance, y_pos + 0.12), shape='dot')
            else:
                G.add_node(current_node_name, pos=(layer_distance, y_pos))
                current_layer_node_names_for_edges.append(current_node_name)

                # Connect Edges
                for prev_node_name in prev_layer_node_names_for_edges:
                     prev_node_actual_idx = -1 # Reset index
                     try:
                         parts = prev_node_name.split('_')
                         if parts[-2] == 'node':
                             prev_node_actual_idx = int(parts[-1])
                     except (IndexError, ValueError):
                         pass # Keep index as -1 if parsing fails
                     if prev_node_actual_idx == -1:
                         continue # Skip connection if previous node index invalid

                     if 0 <= prev_node_actual_idx < num_prev_nodes_actual and 0 <= node_actual_idx < num_nodes_actual:
                         weight_value = value['kernel'][prev_node_actual_idx][node_actual_idx]
                         G.add_edge(prev_node_name, current_node_name, weight=weight_value)
                         if abs(float(weight_value)) > WEIGHT_LABEL_THRESHOLD:
                             edge_weights_dict[(prev_node_name, current_node_name)] = f"{weight_value:.2f}"

        prev_layer_node_names_for_edges = current_layer_node_names_for_edges
        layer_distance += 0.5
    # --- End Node/Edge Creation ---

    # --- Plotting ---
    plt.figure(figsize=(15, 10))
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
         plt.title("Graph Skipped (Empty Positions)")
         plt.close()
         return

    # Text Positioning
    count = 1
    max_y = 0
    min_y = 0
    if pos:
        y_coords = [p[1] for p in pos.values()]
        max_y = max(y_coords) if y_coords else 0
        min_y = min(y_coords) if y_coords else 0
    height_text = max_y + 2.0 if y_coords else 2.0
    activation_text_height = height_text - 0.5

    layer_items_for_text = list(param_dict.items())
    for i, (layer, value) in enumerate(layer_items_for_text):
         if 'kernel' not in value: continue
         num_nodes_actual = value['kernel'].shape[1]
         current_layer_x = 0.5 * (i + 1) # Layer's x position
         if i < len(layer_items_for_text) - 1: # Hidden Layers
            plt.text(current_layer_x, height_text, f"HL {count}", fontsize=11, color='black', ha='center', va='center')
            plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')
            if i < len(activations_list): # Check index bounds for safety
                activation_name = activations_list[i].__name__
                plt.text(current_layer_x, activation_text_height, f"act: {activation_name}", fontsize=9, color='blue', ha='center', va='center')
            count += 1
         else: # Output Layer
             plt.text(current_layer_x, height_text, f"Output Layer", fontsize=11, color='black', ha='center', va='center')
             plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')

    input_layer_x = 0
    plt.text(input_layer_x, height_text, f"Input (GC)", fontsize=11, color='black', ha='center', va='center')
    plt.text(input_layer_x, height_text - 0.25, f"nodes: {num_gc_inputs}", fontsize=9, color='black', ha='center', va='center')

    # Drawing Nodes and Edges
    edge_weights_vals = [abs(d.get('weight', 0)) for u, v, d in G.edges(data=True)]
    max_edge_width = max(edge_weights_vals) if edge_weights_vals else 1
    edge_widths_normalized = [(abs(w) / max_edge_width * 2.5 + 0.2) if max_edge_width > 0 else 0.2 for w in edge_weights_vals]

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6', nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') != 'dot'])
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='grey', nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') == 'dot'])
    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized, alpha=0.5, node_size=400)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights_dict, font_size=7, font_color='red', label_pos=0.3, bbox=dict(alpha=0)) # Use label_pos=0.3

    plt.tight_layout()
    plt.axis('off')
    # --- End Plotting ---


# --- Utility Functions ---
def numpy_collate(batch):
    """Collate function for PyTorch DataLoader."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        return [numpy_collate(samples) for samples in zip(*batch)]
    else:
        return np.array(batch)

def convert_params_to_json_serializable(params):
    """Recursively converts JAX/NumPy arrays in params dict to lists."""
    if isinstance(params, (jnp.ndarray, np.ndarray)):
        return params.tolist()
    elif isinstance(params, (dict, flax.core.FrozenDict)):
        return {k: convert_params_to_json_serializable(v) for k, v in params.items()}
    elif isinstance(params, (list, tuple)):
        return [convert_params_to_json_serializable(item) for item in params]
    elif isinstance(params, (int, float, str, bool, type(None))):
        return params
    else:
        print(f"Warning: Converting unrecognized type {type(params)} to string for JSON.")
        return str(params)


# --- Main Execution Logic ---

@hydra.main(version_base=None, config_path="../task", config_name="config_language") # Assumes config in same dir
def main(cfg: DictConfig): # Use DictConfig type hint

    # --- Setup Output Dirs ---
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output Directory: {output_dir}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- RNG & Initial Config ---
    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.lr

    # --- Dataset Loading (Language Only) ---
    print("Loading Language Dataset")
    vocab = None
    vocab_file = os.path.join(output_dir,"vocab.pkl")
    inv_vocab_file = os.path.join(output_dir,"inv_vocab.pkl")

    try:
        # Ensure text file path exists relative to the original script location if needed
        # Or use hydra's utils to get the original cwd if necessary
        text_file_path = cfg.dataset.text_file
        if not os.path.isabs(text_file_path):
             # Example: Assume it's relative to the original script directory if using hydra's CWD management
             # This might need adjustment depending on how you run hydra and where data lives
             orig_cwd = hydra.utils.get_original_cwd()
             text_file_path = os.path.join(orig_cwd, text_file_path)


        print(f"Attempting to load text from: {text_file_path}")
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Text file not found: {text_file_path}. Please create it or change config.")
    except Exception as e:
        print(f"Error accessing original CWD or text file: {e}")
        raise e


    cleaned_text = clean_text(text)
    vocab, inv_vocab = build_vocab(cleaned_text) # Gets vocab and inverse mapping

    # Update config dynamically - Hydra config is mutable by default within run
    OmegaConf.set_struct(cfg, False) # Allow adding new keys or modifying structure if needed
    cfg.dataset.vocab_size = len(vocab)
    cfg.network.num_output = cfg.dataset.vocab_size
    cfg.network.num_inputs = cfg.dataset.context_window * cfg.dataset.embedding_dim
    OmegaConf.set_struct(cfg, True) # Optional: Lock structure again
    print(f"Updated Config: vocab_size={cfg.dataset.vocab_size}, num_inputs={cfg.network.num_inputs}")

    # Save vocab & inverse vocab
    with open(vocab_file, 'wb') as f: pickle.dump(vocab, f)
    with open(inv_vocab_file, 'wb') as f: pickle.dump(inv_vocab, f)
    print(f"Saved vocabulary to {vocab_file} and {inv_vocab_file}")

    sequences, targets = create_sequences(cleaned_text, vocab, cfg.dataset.context_window)
    if sequences.shape[0] == 0:
         raise ValueError("No sequences generated. Check text file length and context window size.")

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=cfg.dataset.test_split, random_state=cfg.jax.PRNGKey, shuffle=True
    )
    train_dataset = LanguageDataset(X_train, y_train, cfg.dataset.vocab_size)
    test_dataset = LanguageDataset(X_test, y_test, cfg.dataset.vocab_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False)
    print(f"Done. Train sequences: {len(y_train)}, Test sequences: {len(y_test)}")

    # --- Genome representation ---
    current_layer_sizes = list(cfg.network.num_layers) # For GenomeClassifier
    current_activations = []

    # --- Initial Model Creation ---
    rng, model_rng = jax.random.split(rng)
    # Create the initial GC layers/activations definition
    gc_layer_defs, current_activations = create_layers(
        model_rng, current_layer_sizes, cfg.network.num_output, None
    )
    # Instantiate the GC module
    genome_classifier_instance = GenomeClassifier(
        layer_definitions=gc_layer_defs,
        activation_fns=current_activations
    )
    # Create the wrapper model instance
    model_to_init = LanguageModelWrapper(
        genome_classifier=genome_classifier_instance,
        vocab_size=cfg.dataset.vocab_size,
        embedding_dim=cfg.dataset.embedding_dim
    )
    # Prepare dummy input for initialization
    dummy_input_shape = (1, cfg.dataset.context_window)
    dummy_input = jnp.zeros(dummy_input_shape, dtype=jnp.int32)

    # Initialize parameters for the whole wrapper model
    rng, init_rng = jax.random.split(rng)
    params = model_to_init.init(init_rng, dummy_input) # Contains Embed and GC params
    print("Initial model parameters structure:", jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else x, params)) # Use tree_util for deprecation warning

    optimizer = optax.sgd(learning_rate=learning_rate)

    # --- Accuracy Tracking & Detailed Log ---
    accuracy_file = os.path.join(results_dir, "generation_summary.csv")
    detailed_log_file = os.path.join(results_dir, "detailed_log.json")
    detailed_log_data = []
    with open(accuracy_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "TestAccuracy", "NumHiddenLayers", "HiddenLayerSizes", "ActivationFunctions"])

    # --- Generation Loop ---
    for generation in range(cfg.training.generations):
        print(f"\n--- Starting Generation {generation} ---")
        print(f"Current GC Structure: Layers={current_layer_sizes}, Activations={[a.__name__ for a in current_activations]}")

        # --- Create TrainState ---
        # apply_fn is the wrapper model's apply method
        model_state = train_state.TrainState.create(apply_fn=model_to_init.apply, params=params, tx=optimizer)

        # --- Train ---
        trained_model_state, gen_test_accuracy = train_model(
            state=model_state, train_data_loader=train_loader, test_data_loader=test_loader,
            num_epochs=cfg.training.num_epochs, generation=generation, num_output=cfg.network.num_output
        )

        # --- Store Accuracy / Log ---
        with open(accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([ generation, f"{gen_test_accuracy:.6f}", len(current_layer_sizes), str(current_layer_sizes), str([a.__name__ for a in current_activations]) ])

        try:
            state_dict_params = serialization.to_state_dict(trained_model_state.params)
            gc_key = 'genome_classifier' # Use exact key
            gc_params = state_dict_params['params'].get(gc_key, {}) # Get GC dict directly
            serializable_params = convert_params_to_json_serializable(gc_params)
            gen_data = { "generation": generation, "accuracy": float(gen_test_accuracy),
                         "structure": { "input_size": cfg.network.num_inputs, "output_size": cfg.network.num_output,
                                        "hidden_layer_sizes": current_layer_sizes, "activation_functions": [act.__name__ for act in current_activations],
                                        "parameters": serializable_params }}
            detailed_log_data.append(gen_data)
        except Exception as e:
            print(f"Error preparing detailed log data for gen {generation}: {e}")

        # --- Checkpoint Saving ---
        try:
            checkpoints.save_checkpoint( ckpt_dir=checkpoint_dir, target=trained_model_state, step=generation, prefix=f'model_gen_{generation}_', overwrite=True)
            print(f"Saved checkpoint for generation {generation} to {checkpoint_dir}")
        except Exception as e:
            print(f"Error saving checkpoint for gen {generation}: {e}")

        # --- Drawing graph ---
        if cfg.utils.draw_graph:
             gc_key = 'genome_classifier'
             if gc_key in trained_model_state.params['params']:
                 # Extract the GC params dict directly
                 gc_params_dict_for_draw = trained_model_state.params['params'][gc_key]
                 try:
                    draw_graph(gc_params_dict_for_draw, current_activations, cfg) # Pass GC params dict
                    graph_filename = os.path.join(results_dir, f"graph_gen{generation}.png")
                    plt.savefig(graph_filename)
                    plt.close()
                    print(f"Saved graph for generation {generation} to {graph_filename}")
                 except Exception as e:
                     print(f"Error drawing graph for gen {generation}: {e}")
                     plt.close()
             else:
                 print(f"Skipping graph gen {generation}: GC params key '{gc_key}' not found.")

        # --- Evolution / Mutation ---
        print("\n--- Applying Mutations ---")
        rng, evo_rng = jax.random.split(rng)
        mutation_rngs = jax.random.split(evo_rng, 5)

        gc_key = 'genome_classifier'
        if gc_key not in trained_model_state.params['params']:
             keys_found = list(trained_model_state.params['params'].keys())
             raise ValueError(f"Key '{gc_key}' not found in state params for mutation. Keys present: {keys_found}")
        # Extract the GC params dict directly {'layer_definitions_0': ...}
        params_for_mutation = trained_model_state.params['params'][gc_key]

        # Track the next state of the GC part
        next_gc_instance = genome_classifier_instance
        next_gc_params_dict = params_for_mutation # This is now the unwrapped dict
        next_layer_sizes = current_layer_sizes
        next_activations = current_activations
        mutation_applied = False

        # Apply mutations sequentially
        if jax.random.uniform(mutation_rngs[0]) < cfg.neat.add_layer:
            print("Attempting Add Layer (to GC)...")
            rng, mutate_rng = jax.random.split(rng)
            # Expects unwrapped source_gc_params, returns unwrapped new_gc_params_dict
            (gc_instance, gc_params_dict), ls, acts = add_new_layer(mutate_rng, next_layer_sizes, next_activations, next_gc_params_dict, cfg)
            next_gc_instance, next_gc_params_dict, next_layer_sizes, next_activations = gc_instance, gc_params_dict, ls, acts
            mutation_applied = True

        if jax.random.uniform(mutation_rngs[1]) < cfg.neat.add_node:
            print("Attempting Add Node (to GC)...")
            rng, mutate_rng = jax.random.split(rng)
            (gc_instance, gc_params_dict), ls, acts = add_new_node(mutate_rng, next_layer_sizes, next_activations, next_gc_params_dict, cfg)
            next_gc_instance, next_gc_params_dict, next_layer_sizes, next_activations = gc_instance, gc_params_dict, ls, acts
            mutation_applied = True

        if jax.random.uniform(mutation_rngs[2]) < cfg.neat.remove_node:
            print("Attempting Remove Node (from GC)...")
            rng, mutate_rng = jax.random.split(rng)
            (gc_instance, gc_params_dict), ls, acts = remove_node(mutate_rng, next_layer_sizes, next_activations, next_gc_params_dict, cfg)
            next_gc_instance, next_gc_params_dict, next_layer_sizes, next_activations = gc_instance, gc_params_dict, ls, acts
            mutation_applied = True

        if jax.random.uniform(mutation_rngs[3]) < cfg.neat.remove_layer:
            print("Attempting Remove Layer (from GC)...")
            rng, mutate_rng = jax.random.split(rng)
            (gc_instance, gc_params_dict), ls, acts = remove_layer(mutate_rng, next_layer_sizes, next_activations, next_gc_params_dict, cfg)
            next_gc_instance, next_gc_params_dict, next_layer_sizes, next_activations = gc_instance, gc_params_dict, ls, acts
            mutation_applied = True

        if jax.random.uniform(mutation_rngs[4]) < cfg.neat.mutate_activation:
             print("Attempting Mutate Activation (in GC)...")
             rng, mutate_rng = jax.random.split(rng)
             (gc_instance, gc_params_dict), ls, acts = mutate_activation(mutate_rng, next_layer_sizes, next_activations, next_gc_params_dict, cfg)
             next_gc_instance, next_gc_params_dict, next_layer_sizes, next_activations = gc_instance, gc_params_dict, ls, acts
             mutation_applied = True

        # --- Update state for the next generation ---
        current_layer_sizes = next_layer_sizes
        current_activations = next_activations
        genome_classifier_instance = next_gc_instance # Update GC instance reference

        # Rebuild the *entire* parameter dictionary for the TrainState
        embed_key = 'embed'
        gc_key = 'genome_classifier'
        if embed_key not in trained_model_state.params['params']:
            keys_found = list(trained_model_state.params['params'].keys())
            raise ValueError(f"Key '{embed_key}' not found in state params for reconstruction. Keys present: {keys_found}")
        embed_params = trained_model_state.params['params'][embed_key] # Get old Embed params

        # next_gc_params_dict is the dict {'layer_definitions_0': ...}
        params = flax.core.freeze({
            'params': {
                embed_key: embed_params,           # Keep old embed params
                gc_key: next_gc_params_dict        # Insert new GC params dict directly
            }
        })

        # Update model_to_init to use the potentially new GC instance for the next gen's apply_fn
        model_to_init = LanguageModelWrapper(
                genome_classifier=genome_classifier_instance,
                vocab_size=cfg.dataset.vocab_size,
                embedding_dim=cfg.dataset.embedding_dim
        )


    # --- End of Training ---
    print("\n--- Training Complete ---")
    print(f"Simple summary saved to: {accuracy_file}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Graphs saved in: {results_dir}")

    # Save Detailed Log File
    try:
        with open(detailed_log_file, 'w') as f:
            json.dump(detailed_log_data, f, indent=4) # Use indent for readability
        print(f"Detailed structure and accuracy log saved to: {detailed_log_file}")
    except Exception as e:
        print(f"Error saving detailed JSON log: {e}")

    # Plotting Accuracy
    try:
        generations_from_csv, accuracies_from_csv = [], []
        if os.path.exists(accuracy_file):
             with open(accuracy_file, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader:
                    generations_from_csv.append(int(row[0]))
                    accuracies_from_csv.append(float(row[1]))
        if generations_from_csv: # Only plot if data exists
             plt.figure(figsize=(10, 5))
             plt.plot(generations_from_csv, accuracies_from_csv, marker='o')
             plt.title("Test Accuracy over Generations")
             plt.xlabel("Generation")
             plt.ylabel("Test Accuracy")
             plt.grid(True)
             plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))
             plt.close()
             print(f"Accuracy plot saved to: {os.path.join(results_dir, 'accuracy_plot.png')}")
        else:
            print("No accuracy data found in CSV to plot.")
    except Exception as e:
        print(f"Error generating accuracy plot from CSV: {e}")


if __name__ == "__main__":
    # Before running, ensure:
    # 1. `config_language.yaml` exists in the same directory (or adjust config_path).
    # 2. The text file path in `config_language.yaml` (`dataset.text_file`) is correct.
    # 3. Necessary directories (like where the text file resides) exist.
    main()