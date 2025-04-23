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
import os # Added for path manipulation
import csv # Added for saving accuracy


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
    # Handle basic types that are already JSON serializable
    elif isinstance(params, (int, float, str, bool, type(None))):
        return params
    else:
        # Attempt to convert other types to string as a fallback
        print(f"Warning: Converting unrecognized type {type(params)} to string for JSON.")
        return str(params)

def draw_graph(trained_model_state, activations_list, cfg): # Added activations_list
    """
    Draw a graph representing the trained model architecture, weights, and activations.

    Parameters:
    trained_model_state: Trained model state (flax.training.train_state.TrainState).
    activations_list: List of activation functions used for hidden layers.
    cfg: Configuration object containing network parameters.
    """
    G = nx.DiGraph()

    # --- Parameter Check (keep the improved check from previous step) ---
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
    # --- End Parameter Check ---

    # --- Node and Edge Creation (Keep improved ellipsis handling) ---
    # (Code for adding input nodes, hidden layer nodes, output layer nodes, and edges)
    # ... (Assume the improved node/edge creation logic with ellipsis handling is here) ...
    # Important: Ensure G.add_edge includes the weight attribute:
    # G.add_edge(prev_node_name, current_node_name, weight=weight_value)
    # --- (Previous node/edge creation code goes here) ---
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

    prev_layer_node_names_for_edges = [n for n in start_nodes if 'ellipsis' not in n] # Actual nodes for edges
    prev_layer_nodes_vis = start_nodes # Nodes including ellipsis for positioning logic

    layer_distance = 0.5
    max_nodes = 0
    layer_items = list(param_dict.items())

    # Dictionary to store edge weights for labeling
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
        logical_vis_count = num_nodes_actual # Count for positioning if no ellipsis
        if use_ellipsis_current:
            vis_indices_current = list(range(3)) + [-1] + list(range(num_nodes_actual - 3, num_nodes_actual))
            logical_vis_count = 7 # 3 + ellipsis + 3 for positioning

        current_layer_node_names_for_edges = [] # Store actual nodes for next layer edges
        current_layer_nodes_vis = [] # Store all visualized nodes (incl ellipsis)

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

                # --- Connect Edges (Simplified for Ellipsis) ---
                for prev_node_name in prev_layer_node_names_for_edges: # Iterate only over actual previous nodes
                     # Map prev_node_name back to its actual index (this assumes names are like 'layer_node_idx')
                     try:
                         parts = prev_node_name.split('_')
                         if parts[-2] == 'node': # Check name format
                             prev_node_actual_idx = int(parts[-1])
                         else: continue # Skip if name format is unexpected
                     except (IndexError, ValueError):
                         continue # Skip if parsing fails

                     if 0 <= prev_node_actual_idx < num_prev_nodes_actual and 0 <= node_actual_idx < num_nodes_actual:
                         weight_value = value['kernel'][prev_node_actual_idx][node_actual_idx]
                         G.add_edge(prev_node_name, current_node_name, weight=weight_value)
                         # Store weight for label, format it
                         edge_weights_dict[(prev_node_name, current_node_name)] = f"{weight_value:.2f}"


        prev_layer_node_names_for_edges = current_layer_node_names_for_edges # Update actual nodes for next iteration
        prev_layer_nodes_vis = current_layer_nodes_vis # Update visualized nodes
        layer_distance += 0.5
    # --- End Node and Edge Creation ---


    # Plotting
    plt.figure(figsize=(15, 10)) # Increase figure size slightly
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
         print("Warning: Graph position attribute is empty. Skipping draw.")
         plt.title("Graph Generation Skipped (Empty Positions)")
         plt.close()
         return

    # --- Text Positioning ---
    distance = 0.5
    count = 1
    max_y = 0
    min_y = 0
    if pos:
        y_coords = [p[1] for p in pos.values()]
        if y_coords:
            max_y = max(y_coords)
            min_y = min(y_coords)
    # Adjust height based on graph extent, add more space
    height_text = max_y + 2.0 if y_coords else 2.0
    activation_text_height = height_text - 0.5 # Position activation below layer info

    layer_items_for_text = list(param_dict.items())

    for i, (layer, value) in enumerate(layer_items_for_text):
         if 'kernel' not in value: continue
         num_nodes_actual = value['kernel'].shape[1]
         current_layer_x = 0.5 * (i + 1)

         if i < len(layer_items_for_text) - 1: # Hidden Layers
            plt.text(current_layer_x, height_text, f"HL {count}", fontsize=11, color='black', ha='center', va='center')
            plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')
            # --- Add Activation Text ---
            if i < len(activations_list): # Check index bounds
                activation_name = activations_list[i].__name__
                plt.text(current_layer_x, activation_text_height, f"act: {activation_name}", fontsize=9, color='blue', ha='center', va='center')
            # --- End Activation Text ---
            count += 1
         else: # Output Layer
             plt.text(current_layer_x, height_text, f"Output Layer", fontsize=11, color='black', ha='center', va='center')
             plt.text(current_layer_x, height_text - 0.25, f"nodes: {num_nodes_actual}", fontsize=9, color='black', ha='center', va='center')
             # Output layer typically doesn't have an activation shown here (it's often in the loss)

    # Input Layer Text
    input_layer_x = 0
    plt.text(input_layer_x, height_text, f"Input Layer", fontsize=11, color='black', ha='center', va='center')
    plt.text(input_layer_x, height_text - 0.25, f"nodes: {cfg.network.num_inputs}", fontsize=9, color='black', ha='center', va='center')
    # --- End Text Positioning ---

    # --- Drawing Nodes and Edges ---
    edge_weights_vals = [abs(d.get('weight', 0)) for u, v, d in G.edges(data=True)]
    max_edge_width = max(edge_weights_vals) if edge_weights_vals else 1
    # Make edge width scaling less extreme, add minimum width
    edge_widths_normalized = [(abs(w) / max_edge_width * 2.5 + 0.2) if max_edge_width > 0 else 0.2 for w in edge_weights_vals]

    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='#ADD8E6', # Smaller nodes
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') != 'dot'])
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color='grey', # Smaller ellipsis dots
            nodelist=[n for n, d in G.nodes(data=True) if d.get('shape') == 'dot'])
    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized, alpha=0.5, node_size=400) # Pass node_size to edges for arrow placement

    # --- Add Edge Labels (Weights) ---
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_weights_dict,
        font_size=7,
        font_color='red',
        label_pos=0.1,  # <--- CHANGE: Shift position along edge (try 0.3 or 0.7)
        bbox=dict(alpha=0) # Keep background transparent
    )
    # --- End Edge Labels ---

    plt.tight_layout() # Adjust layout
    plt.axis('off') # Turn off axis

# --- calculate_loss_acc function remains the same ---
def calculate_loss_acc(state, params, batch, num_output):
    """
    Calculate loss and accuracy for a batch of data.

    Parameters:
    state: Model state.
    params: Model parameters.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    tuple: Tuple containing loss and accuracy.
    """
    data_input, labels = batch
    # Ensure params are correctly accessed if nested (e.g., state.params['params'])
    model_params = params['params'] if 'params' in params else params

    try:
        logits = state.apply_fn({'params': model_params}, data_input) # Pass params in the expected structure
        loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

        # Logits to probabilities
        probs = jax.nn.softmax(logits)
        max_index = jnp.argmax(probs, axis=-1)

        pred_labels = jax.nn.one_hot(max_index, num_output)

        # Ensure labels are one-hot encoded for comparison if they aren't already
        # (This depends on how labels are loaded, assuming they are one-hot based on Iris/DigitsDataset)
        acc = jnp.all(pred_labels == labels, axis=-1).mean()

    except Exception as e:
        print(f"Error during loss/acc calculation: {e}")
        print(f"Input shape: {data_input.shape}, Labels shape: {labels.shape}")
        # print(f"Model Params Structure: {jax.tree_util.tree_map(lambda x: x.shape, model_params)}")
        # Consider returning default values or raising the exception
        return jnp.inf, 0.0 # Return infinity loss and 0 accuracy on error


    return loss, acc

# --- train_step remains the same ---
@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, num_output):
    """
    Perform a single training step.

    Parameters:
    state: Model state.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    tuple: Tuple containing updated model state, loss, and accuracy.
    """
    # Gradient function
    grad_fn = jax.value_and_grad(calculate_loss_acc,  # Function to calculate the loss
                                 argnums=1,  # Parameters are second argument of the function (params)
                                 has_aux=True  # Function has additional outputs, here accuracy
                                 )
    # Determine gradients for current model, parameters and batch
    # Pass state.params directly as the second argument
    (loss, acc), grads = grad_fn(state, state.params, batch, num_output)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss, acc


def train_model(state, train_data_loader, test_data_loader, num_epochs, generation, num_output):
    """
    Train the model for a specified number of epochs.

    Parameters:
    state: Initial model state.
    train_data_loader: Data loader for training dataset.
    test_data_loader: Data loader for testing dataset.
    num_epochs (int): Number of epochs to train.
    generation (int): Current generation number.
    num_output (int): Number of output classes.

    Returns:
        tuple: (trained_model_state, final_test_accuracy)
    """
    best_epoch_acc = -1.0
    for epoch in tqdm(range(num_epochs), desc=f"Gen {generation} Training"):
        batch_losses = []
        batch_accs = []
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch, num_output)
            batch_losses.append(loss)
            batch_accs.append(acc)

        # --- Evaluate on test set periodically (e.g., every epoch) ---
        epoch_test_acc = eval_model(state, test_data_loader, num_output)
        # Optional: Log epoch-level performance
        # train_loss_epoch = np.mean(batch_losses)
        # train_acc_epoch = np.mean(batch_accs)
        # print(f"  Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.4f} | Test Acc: {epoch_test_acc:.4f}")
        best_epoch_acc = max(best_epoch_acc, epoch_test_acc)


    # --- Final evaluation after all epochs for this generation ---
    final_test_acc = eval_model(state, test_data_loader, num_output)
    print(f"Generation {generation} Final Test Accuracy: {final_test_acc:.4f} (Best during epoch: {best_epoch_acc:.4f})")

    return state, final_test_acc # Return state and the final accuracy


# --- eval_step remains the same ---
@partial(jax.jit, static_argnums=(2,))
def eval_step(state, batch, num_output):
    """
    Evaluate the model on a single batch.

    Parameters:
    state: Model state.
    batch: Tuple containing input data and labels.
    num_output (int): Number of output classes.

    Returns:
    float: Accuracy.
    """
    # Determine the accuracy
    _, acc = calculate_loss_acc(state, state.params, batch, num_output)
    return acc


# --- Modified eval_model to return accuracy ---
def eval_model(state, data_loader, num_output):
    """
    Evaluate the model on the entire dataset.

    Parameters:
    state: Model state.
    data_loader: Data loader for evaluation dataset.
    num_output (int): Number of output classes.

    Returns:
        float: Overall accuracy on the dataset.
    """
    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch, num_output)
        all_accs.append(batch_acc)
        # Ensure batch elements are numpy arrays for shape access if coming from torch DataLoader
        input_data = batch[0]
        if isinstance(input_data, torch.Tensor):
             input_data = input_data.numpy() # Convert if necessary
        batch_sizes.append(input_data.shape[0])

    # Weighted average since some batches might be smaller
    # Handle cases where data_loader might be empty
    if not batch_sizes:
        print("Warning: Evaluation data loader is empty.")
        return 0.0
    if sum(batch_sizes) == 0:
         print("Warning: Total batch size is zero during evaluation.")
         return 0.0

    acc = sum([a * b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    return float(acc) # Return the calculated accuracy


# --- numpy_collate, Datasets, mnist_transform, mnist_collate_fn remain the same ---
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

# --- Added list of possible activations ---
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

# --- Modified create_layers to handle activation list ---
def create_layers(rng, layer_sizes, num_output, current_activations=None):
    """
    Create layers and corresponding activations for the model.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    layer_sizes (list): List of integers representing the number of hidden units in each layer.
    num_output (int): Number of output units in the final layer.
    current_activations (list or None): List of activation functions to use. If None or length
                                         mismatch, new ones are chosen randomly.

    Returns:
    tuple: Tuple containing lists of layers and activation functions.
    """
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
            # --- FIX START ---
            rng, act_rng = jax.random.split(rng)
            # Choose a random index into the list of activation functions
            num_possible_activations = len(POSSIBLE_ACTIVATION_FNS)
            indices_array = jnp.arange(num_possible_activations)
            chosen_index = jax.random.choice(act_rng, indices_array)
            # Use the index to get the actual function from the Python list
            chosen_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
            # --- FIX END ---
            activations.append(chosen_activation)
        else:
            # Use the provided activation
            activations.append(current_activations[i])

    # Output layer
    layers.append(nn.Dense(features=num_output))

    return layers, activations



class GenomeClassifier(nn.Module):
    layer_definitions: list # List of layer objects (e.g., nn.Dense instances)
    activation_fns: list    # List of activation functions (callable)

    @nn.compact
    def __call__(self, x):
        if len(self.layer_definitions) != len(self.activation_fns) + 1:
             raise ValueError(f"Mismatch between layers ({len(self.layer_definitions)}) and activations ({len(self.activation_fns)})")

        # Apply hidden layers and activations
        for i in range(len(self.activation_fns)):
            x = self.layer_definitions[i](x)
            x = self.activation_fns[i](x) # Apply corresponding activation

        # Apply final output layer (no activation here, typically softmax is in loss)
        x = self.layer_definitions[-1](x)
        return x


class SimpleClassifier(nn.Module): # Keep for reference or remove if unused
    num_hidden: int
    num_outputs: int

    @nn.compact # Use compact for automatic setup
    def __call__(self, x):
        x = nn.Dense(features=self.num_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x


# --- Modified copy_layers_over to handle activations ---
# --- Modified copy_params_and_create_model to handle activations ---
def copy_params_and_create_model(rng, layer_sizes, current_activations, num_output, num_inputs, source_params=None):
    """
    Creates a new model instance and copies parameters from a source, handling size mismatches.
    ... (docstring remains the same) ...
    """
    rng, layer_rng, init_rng, input_rng = jax.random.split(rng, 4)

    layers, activations = create_layers(layer_rng, layer_sizes, num_output, current_activations)
    new_model = GenomeClassifier(layer_definitions=layers, activation_fns=activations)

    # Initialize new model to get the target shape
    dummy_input = jax.random.normal(input_rng, (num_inputs,))
    target_params = new_model.init(init_rng, dummy_input)['params']
    new_params = target_params # Start with newly initialized params

    if source_params is None:
        print("No source parameters provided, returning newly initialized model.")
        return new_model, {'params': new_params} # Ensure structure matches TrainState

    # --- Parameter Copying Logic ---
    new_params_mutable = flax.core.unfreeze(new_params)
    # Ensure source_params is a standard dict if it's a FrozenDict for easier access
    if isinstance(source_params, flax.core.FrozenDict):
        source_params_dict = source_params.unfreeze()
    else:
        # Assume it might already be nested under 'params' if coming from TrainState
        source_params_dict = source_params.get('params', source_params)
        if isinstance(source_params_dict, flax.core.FrozenDict):
             source_params_dict = source_params_dict.unfreeze()


    source_params_layers = list(source_params_dict.items())
    target_params_layers = list(new_params_mutable.items()) # target is already mutable dict

    num_layers_to_copy = min(len(source_params_layers), len(target_params_layers))

    for i in range(num_layers_to_copy):
        source_layer_name, source_layer_values = source_params_layers[i]
        target_layer_name, target_layer_values = target_params_layers[i]

        print(f"Copying layer {i}: {source_layer_name} -> {target_layer_name}")

        if 'kernel' in source_layer_values and 'kernel' in target_layer_values:
            # --- FIX: Work directly with JAX arrays ---
            # Remove jax.device_get here
            source_kernel = source_layer_values['kernel'] # This should be a JAX array
            target_kernel = target_layer_values['kernel'] # This should be a JAX array

            # Check if they are indeed JAX arrays (optional sanity check)
            # assert isinstance(source_kernel, jnp.ndarray), f"Expected JAX array, got {type(source_kernel)}"
            # assert isinstance(target_kernel, jnp.ndarray), f"Expected JAX array, got {type(target_kernel)}"

            # Determine copy dimensions
            copy_rows = min(source_kernel.shape[0], target_kernel.shape[0])
            copy_cols = min(source_kernel.shape[1], target_kernel.shape[1])

            # Create a copy of the target kernel (this will be a JAX array copy)
            updated_kernel = target_kernel.copy()
            # Copy the overlapping part from the source using JAX indexed update
            # This should now work because updated_kernel is a JAX array
            updated_kernel = updated_kernel.at[:copy_rows, :copy_cols].set(source_kernel[:copy_rows, :copy_cols])

            new_params_mutable[target_layer_name]['kernel'] = updated_kernel # Assign the updated JAX array
            # print(f"  Kernel copied: {source_kernel.shape} ({copy_rows}x{copy_cols}) -> {target_kernel.shape}")


        if 'bias' in source_layer_values and 'bias' in target_layer_values:
            # --- FIX: Work directly with JAX arrays ---
            # Remove jax.device_get here
            source_bias = source_layer_values['bias'] # JAX array
            target_bias = target_layer_values['bias'] # JAX array

            copy_len = min(len(source_bias), len(target_bias))

            # Create a JAX array copy
            updated_bias = target_bias.copy()
            # Use JAX indexed update
            updated_bias = updated_bias.at[:copy_len].set(source_bias[:copy_len])

            new_params_mutable[target_layer_name]['bias'] = updated_bias # Assign updated JAX array
            # print(f"  Bias copied: {source_bias.shape} ({copy_len}) -> {target_bias.shape}")


    return new_model, {'params': flax.core.freeze(new_params_mutable)} # Re-freeze and ensure structure

# --- NEAT Mutation Functions ---

# --- NEAT Mutation Functions ---

def add_new_layer(rng, layer_sizes, current_activations, source_params, cfg):
    """ Adds a new hidden layer with size 1 and random activation. """
    rng, insert_rng, act_rng = jax.random.split(rng, 3)
    insert_pos = jax.random.randint(insert_rng, (1,), 0, len(layer_sizes) + 1).item() # Position to insert
    new_layer_size = 1 # Start with a small layer

    new_layer_sizes = layer_sizes[:insert_pos] + [new_layer_size] + layer_sizes[insert_pos:]

    # --- FIX START ---
    # Add a random activation for the new layer by choosing an index
    # OLD: new_activation = jax.random.choice(act_rng, jnp.array(POSSIBLE_ACTIVATION_FNS))

    num_possible_activations = len(POSSIBLE_ACTIVATION_FNS)
    indices_array = jnp.arange(num_possible_activations)
    chosen_index = jax.random.choice(act_rng, indices_array)
    # Use the index to get the actual function from the Python list
    new_activation = POSSIBLE_ACTIVATION_FNS[chosen_index]
    # --- FIX END ---

    new_activations = current_activations[:insert_pos] + [new_activation] + current_activations[insert_pos:]

    print(f"  New Layer Sizes: {new_layer_sizes}")
    print(f"  New Activations: {[a.__name__ for a in new_activations]}") # Print names

    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations

# --- (Rest of the mutation functions: add_new_node, remove_node, remove_layer, mutate_activation) ---
# Ensure the fixes applied previously are still in place in create_layers and mutate_activation

def add_new_node(rng, layer_sizes, current_activations, source_params, cfg):
    """ Adds a new node to a randomly selected existing hidden layer. """
    if not layer_sizes: # Cannot add node if no hidden layers exist
        print("  Skipping Add Node: No hidden layers exist.")
        # Optionally, could call add_new_layer here
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations

    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.randint(choice_rng, (1,), 0, len(layer_sizes)).item()

    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] += 1 # Increment node count

    print(f"  Incremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")

    # Activations list doesn't change length when adding a node
    new_activations = current_activations

    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations


def remove_node(rng, layer_sizes, current_activations, source_params, cfg):
    """ Removes a node from a randomly selected hidden layer (if possible). """
    # Find layers with more than 1 node (cannot remove the last node)
    eligible_layers_indices = [i for i, size in enumerate(layer_sizes) if size > 1]

    if not eligible_layers_indices:
        print("  Skipping Remove Node: No eligible layers found (all have size 1 or fewer).")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations

    rng, choice_rng = jax.random.split(rng)
    layer_to_modify_idx = jax.random.choice(choice_rng, jnp.array(eligible_layers_indices)).item()

    new_layer_sizes = layer_sizes.copy()
    new_layer_sizes[layer_to_modify_idx] -= 1 # Decrement node count

    print(f"  Decremented layer {layer_to_modify_idx}. New Sizes: {new_layer_sizes}")

    # Activations list doesn't change length when removing a node
    new_activations = current_activations

    # NOTE: copy_params_and_create_model handles the weight copying for the reduced size.
    # It copies the *top-left* portion of weights. More sophisticated removal
    # (e.g., removing the node with least impact) is much more complex.
    return copy_params_and_create_model(rng, new_layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), new_layer_sizes, new_activations


def remove_layer(rng, layer_sizes, current_activations, source_params, cfg):
    """ Removes a randomly selected hidden layer (if any exist). """
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
    """ Mutates the activation function of a randomly selected hidden layer. """
    if not current_activations:
        print("  Skipping Mutate Activation: No hidden layers/activations exist.")
        model, params = copy_params_and_create_model(rng, layer_sizes, current_activations, cfg.network.num_output, cfg.network.num_inputs, source_params)
        return (model, params), layer_sizes, current_activations

    rng, choice_rng, act_rng = jax.random.split(rng, 3)
    layer_to_mutate_idx = jax.random.randint(choice_rng, (1,), 0, len(current_activations)).item()

    current_activation_fn = current_activations[layer_to_mutate_idx]
    # Select a new activation, ensuring it's different from the current one if possible
    possible_new_activations = [fn for fn in POSSIBLE_ACTIVATION_FNS if fn != current_activation_fn]
    if not possible_new_activations: # Only one type of activation available? Use the current one.
        new_activation_fn = current_activation_fn
    else:
        # Create a list of indices corresponding to the allowed new functions
        possible_indices = [i for i, fn in enumerate(POSSIBLE_ACTIVATION_FNS) if fn in possible_new_activations]
        # Choose an index *relative* to the possible_indices list
        chosen_relative_index = jax.random.choice(act_rng, jnp.arange(len(possible_indices)))
        # Get the absolute index in POSSIBLE_ACTIVATION_FNS
        absolute_index = possible_indices[chosen_relative_index]
        # Select the function using the absolute index
        new_activation_fn = POSSIBLE_ACTIVATION_FNS[absolute_index]
    # --- FIX END ---

    new_activations = current_activations.copy()
    new_activations[layer_to_mutate_idx] = new_activation_fn

    # ... (rest of the function) ...
    return copy_params_and_create_model(rng, layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), layer_sizes, new_activations



    # Recreate the model with the same layer sizes but new activations
    # Parameters are copied over.
    return copy_params_and_create_model(rng, layer_sizes, new_activations, cfg.network.num_output, cfg.network.num_inputs, source_params), layer_sizes, new_activations


# @hydra.main(version_base=None, config_path="../task", config_name="config_mnist")
@hydra.main(version_base=None, config_path="../task", config_name="config_iris")
def main(cfg):

    # --- Setup Output Dirs (as before) ---
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Output Directory: {output_dir}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # --- RNG and Initial Config (as before) ---
    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)
    current_layer_sizes = list(cfg.network.num_layers)
    current_activations = []
    num_inputs = cfg.network.num_inputs
    num_output = cfg.network.num_output
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.lr

    # --- Initial Model Creation (as before) ---
    rng, model_rng = jax.random.split(rng)
    _, current_activations = create_layers(model_rng, current_layer_sizes, num_output, None)
    print(f"Initial Hidden Layer Sizes: {current_layer_sizes}")
    print(f"Initial Activations: {[a.__name__ for a in current_activations]}")
    model, params = copy_params_and_create_model(
        model_rng, current_layer_sizes, current_activations, num_output, num_inputs, source_params=None
    )
    optimizer = optax.sgd(learning_rate=learning_rate)

    # --- Dataset Loading (remains the same) ---
    if cfg.dataset.dataset_type == "digits":
        print("Loading Digits Dataset")
        sklearn_dataset = datasets.load_digits()
        n_samples = len(sklearn_dataset.images)
        data = sklearn_dataset.images.reshape((n_samples, -1)) / 16.0 # Normalize
        X_train, X_test, y_train, y_test = train_test_split(data, sklearn_dataset.target, test_size=0.3, shuffle=True, random_state=cfg.jax.PRNGKey)
        train_dataset = DigitsDataset(X_train, y_train, num_classes=num_output)
        test_dataset = DigitsDataset(X_test, y_test, num_classes=num_output)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False) # Don't drop last for test
        print("Done.")
    elif cfg.dataset.dataset_type == "iris":
        print("Loading Iris Dataset")
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        # Normalize features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        # Avoid division by zero if std is 0 for any feature
        X_std = np.where(X_std == 0, 1, X_std)
        X = (X - X_mean) / X_std
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=cfg.jax.PRNGKey, stratify=y) # Stratify for iris
        train_dataset = IrisDataset(X_train, y_train, num_classes=num_output)
        test_dataset = IrisDataset(X_test, y_test, num_classes=num_output)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate, drop_last=False)
        print("Done.")
    elif cfg.dataset.dataset_type == "mnist":
        print("Loading MNIST Dataset")
        # Use PyTorch data loading as before
        train = MNIST(root='train', train=True, transform=mnist_transform, download=True)
        test = MNIST(root='test', train=False, transform=mnist_transform, download=True)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=mnist_collate_fn, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=mnist_collate_fn, drop_last=False)
        print("Done.")
    else:
        raise ValueError(f"Unknown dataset type: {cfg.dataset.dataset_type}")


    # --- Accuracy Tracking & Detailed Log ---
    # generation_accuracies = [] # Keep this if you still want the simple list
    accuracy_file = os.path.join(results_dir, "generation_summary.csv") # Simple summary CSV
    detailed_log_file = os.path.join(results_dir, "detailed_log.json") # Detailed JSON log
    detailed_log_data = [] # List to store data for JSON log

    # --- Generation Loop ---
    for generation in range(cfg.training.generations):
        print(f"\n--- Starting Generation {generation} ---")
        print(
            f"Current Structure: Layers={current_layer_sizes}, Activations={[a.__name__ for a in current_activations]}")

        # --- Create TrainState (as before) ---
        current_params = params if 'params' in params else {'params': params}
        model_state = train_state.TrainState.create(apply_fn=model.apply, params=current_params, tx=optimizer)

        # --- Train the current model (as before) ---
        trained_model_state, gen_test_accuracy = train_model(
            state=model_state,
            train_data_loader=train_loader,
            test_data_loader=test_loader,
            num_epochs=cfg.training.num_epochs,
            generation=generation,
            num_output=num_output
        )

        # --- Store and Save Simple Accuracy Summary ---
        # generation_accuracies.append(gen_test_accuracy) # Append to simple list if needed
        with open(accuracy_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                generation,
                f"{gen_test_accuracy:.6f}",
                len(current_layer_sizes),
                str(current_layer_sizes),
                str([a.__name__ for a in current_activations])
            ])

        # --- Prepare Data for Detailed JSON Log ---
        try:
            # Convert trained parameters to JSON serializable format
            state_dict_params = serialization.to_state_dict(trained_model_state.params)
            serializable_params = convert_params_to_json_serializable(
                state_dict_params.get('params', {}))  # Get inner 'params' dict

            gen_data = {
                "generation": generation,
                "accuracy": float(gen_test_accuracy),  # Ensure accuracy is float
                "structure": {
                    "input_size": cfg.network.num_inputs,
                    "output_size": cfg.network.num_output,
                    "hidden_layer_sizes": current_layer_sizes,
                    "activation_functions": [act.__name__ for act in current_activations],
                    "parameters": serializable_params  # Add serializable params
                }
            }
            detailed_log_data.append(gen_data)
        except Exception as e:
            print(f"Error preparing detailed log data for generation {generation}: {e}")

        # --- Periodic Checkpoint Saving (as before) ---
        # ... (Checkpoint saving code remains the same) ...
        try:
            checkpoints.save_checkpoint(
                ckpt_dir=checkpoint_dir,
                target=trained_model_state,  # Save the entire TrainState
                step=generation,  # Use generation number as step
                prefix=f'model_gen_{generation}_',
                overwrite=True  # Overwrite previous checkpoints for this generation prefix
            )
            print(f"Saved checkpoint for generation {generation} to {checkpoint_dir}")
        except Exception as e:
            print(f"Error saving checkpoint for generation {generation}: {e}")

        # --- Drawing graph (Pass activations) ---
        if cfg.utils.draw_graph:
            try:
                # Pass the current_activations list to draw_graph
                draw_graph(trained_model_state, current_activations, cfg)
                graph_filename = os.path.join(results_dir, f"graph_gen{generation}.png")
                plt.savefig(graph_filename)
                plt.close()
                print(f"Saved graph for generation {generation} to {graph_filename}")
            except Exception as e:
                print(f"Error drawing or saving graph for generation {generation}: {e}")
                plt.close()  # Close plot even if error occurred

        # --- Evolution / Mutation (as before) ---
        # ... (Mutation code remains the same) ...
        print("\n--- Applying Mutations ---")
        rng, evo_rng = jax.random.split(rng)
        mutation_rngs = jax.random.split(evo_rng, 5)  # One key per mutation type
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
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
        # Add new node
        if jax.random.uniform(mutation_rngs[1]) < cfg.neat.add_node:
            print("Attempting Add Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = add_new_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
        # Remove node
        if jax.random.uniform(mutation_rngs[2]) < cfg.neat.remove_node:
            print("Attempting Remove Node...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_node(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
        # Remove layer
        if jax.random.uniform(mutation_rngs[3]) < cfg.neat.remove_layer:
            print("Attempting Remove Layer...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = remove_layer(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )
        # Mutate Activation
        if jax.random.uniform(mutation_rngs[4]) < cfg.neat.mutate_activation:
            print("Attempting Mutate Activation...")
            rng, mutate_rng = jax.random.split(rng)
            (next_model, next_params), next_layer_sizes, next_activations = mutate_activation(
                mutate_rng, next_layer_sizes, next_activations, current_params_for_mutation, cfg
            )

        # Update state for the next generation
        model = next_model
        params = next_params
        current_layer_sizes = next_layer_sizes
        current_activations = next_activations

    # --- End of Training ---
    print("\n--- Training Complete ---")
    # print(f"Final Test Accuracies per Generation: {generation_accuracies}") # Use CSV/JSON instead
    print(f"Simple summary saved to: {accuracy_file}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print(f"Graphs saved in: {results_dir}")

    # --- Save Detailed Log File ---
    try:
        with open(detailed_log_file, 'w') as f:
            json.dump(detailed_log_data, f, indent=4)  # Use indent for readability
        print(f"Detailed structure and accuracy log saved to: {detailed_log_file}")
    except Exception as e:
        print(f"Error saving detailed JSON log: {e}")

    # --- Plotting Accuracy (as before) ---
    # Load accuracy from the CSV for plotting
    try:
        generations_from_csv = []
        accuracies_from_csv = []
        with open(accuracy_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
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
    except Exception as e:
        print(f"Error generating accuracy plot from CSV: {e}")

if __name__ == "__main__":
    main()