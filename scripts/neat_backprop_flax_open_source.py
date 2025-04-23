from functools import partial
import hydra
import flax
from flax.training import train_state, checkpoints
from flax import linen as nn
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
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




def draw_graph(trained_model_state, cfg):
    """
    Draw a graph representing the trained model architecture and weights.

    Parameters:
    trained_model_state: Trained model state.
    cfg: Configuration object containing network parameters.
    """

    G = nx.DiGraph()
    num_elements = cfg.network.num_inputs
    start_nodes = [f'input_node_{i + 1}' for i in range(min(num_elements, 6))]

    for i, node in enumerate(start_nodes):
        y_pos = i - (len(start_nodes) - 1) / 2
        G.add_node(node, pos=(0, y_pos))

    prev_layer_nodes = start_nodes
    layer_distance = 0.5
    max_nodes = 0

    for layer, value in trained_model_state.params['params'].items():
        if 'kernel' in value:
            num_prev_nodes = min(len(value['kernel']), 6)
            num_nodes = min(len(value['kernel'][0]), 6)
            max_nodes = max(max_nodes, num_nodes)

        for node_i, node in enumerate(range(num_nodes)):
            current_node = f"{layer}_node_{node}"
            y_pos = node_i - (num_nodes - 1) / 2
            G.add_node(current_node, pos=(layer_distance, y_pos))

            for weight in range(num_prev_nodes):
                prev_node = prev_layer_nodes[weight]
                weight_value = value['kernel'][weight][node]
                G.add_edge(prev_node, current_node, weight=weight_value)

        prev_layer_nodes = [f"{layer}_node_{node}" for node in range(num_nodes)]

        if len(value['kernel']) > 6:
            # If there are more than 5 nodes in the layer, add three dots in the middle
            num_nodes = 3
            middle_y = (num_prev_nodes - 1) / 2 - 2.5
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y - 0.12), shape='dot')
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y), shape='dot')
            G.add_node(np.random.randint(10000), pos=(layer_distance - 0.5, middle_y + 0.12), shape='dot')

        layer_distance += 0.5

    # Plotting
    # pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    pos = nx.get_node_attributes(G, 'pos')

    distance = 0.5
    count = 1
    height_text = 2
    for i, (layer, value) in enumerate(trained_model_state.params['params'].items()):
        num_nodes = min(len(value['kernel'][0]), 6)

        if num_nodes <= 4:
            height_text = 2
        elif num_nodes == 5:
            height_text = 2.5
        else:
            height_text = 3

        if i < len(trained_model_state.params['params'].items()) - 1:
            plt.text(distance - 0.08, height_text, f"HL {count}", fontsize=12, color='black')
            plt.text(distance - 0.08, height_text - 0.2, f"num nodes {len(value['kernel'].T)}", fontsize=12,
                     color='black')
            count += 1
            distance += 0.5
        else:
            plt.text(distance - 0.07, height_text, f"Output Layer", fontsize=12, color='black')

    plt.text(-0.05, height_text, f"Input Layer", fontsize=12, color='black')

    # Calculate edge widths based on absolute value of weights
    edge_widths = [abs(d['weight']) for u, v, d in G.edges(data=True)]
    edge_widths_normalized = [w / max(edge_widths) * 3 for w in edge_widths]

    nx.draw(G, pos, with_labels=False, node_size=1500, node_color='#C70039', font_size=10, font_weight='bold',
            nodelist=[n for n, d in G.nodes(data=True) if not 'shape' in d])
    nx.draw(G, pos, with_labels=False, node_size=30, node_color='grey',
            nodelist=[n for n, d in G.nodes(data=True) if 'shape' in d and d['shape'] == 'dot'])

    nx.draw_networkx_edges(G, pos, width=edge_widths_normalized)


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

    logits = state.apply_fn(params, data_input)
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

    # Logits to probabilities
    probs = jax.nn.softmax(logits)
    max_index = jnp.argmax(probs, axis=-1)

    pred_labels = jax.nn.one_hot(max_index, num_output)

    acc = jnp.all(pred_labels == labels, axis=-1).mean()

    return loss, acc


# @jax.jit  # Jit the function for efficiency
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
                                 argnums=1,  # Parameters are second argument of the function
                                 has_aux=True  # Function has additional outputs, here accuracy
                                 )
    # Determine gradients for current model, parameters and batch
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
    state: Trained model state.
    """

    for epoch in tqdm(range(num_epochs)):
        batch_loss = []
        batch_acc = []
        for batch in train_data_loader:
            state, loss, acc = train_step(state, batch, num_output)
            batch_loss.append(loss)
            batch_acc.append(acc)

        eval_model(state, test_data_loader, epoch, generation, num_output)

    return state


# @jax.jit  # Jit the function for efficiency
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


def eval_model(state, data_loader, epoch, generation, num_output):
    """
    Evaluate the model on the entire dataset.

    Parameters:
    state: Model state.
    data_loader: Data loader for evaluation dataset.
    epoch (int): Current epoch number.
    generation (int): Current generation number.
    num_output (int): Number of output classes.
    """

    all_accs, batch_sizes = [], []
    for batch in data_loader:
        batch_acc = eval_step(state, batch, num_output)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller
    acc = sum([a * b for a, b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)


# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
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
    """
    Transform MNIST image data.

    Parameters:
    x (array-like): Input image data.

    Returns:
    array-like: Flattened and normalized image data.
    """
    np_img = np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.
    return np_img.flatten()

def mnist_collate_fn(batch):
    """
    Collate function for MNIST dataset.

    Parameters:
    batch (list): List of tuples, each containing input image and corresponding label.

    Returns:
    tuple: Tuple of two arrays - input images and one-hot encoded labels.
    """
    batch = list(zip(*batch))
    x = np.stack(batch[0])
    y = jax.nn.one_hot(np.array(batch[1]), 10)
    return x, y


def create_layers(rng, num_hidden, num_output, prev_activations):
    """
    Create layers and corresponding activations for the model.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_hidden (list): List of integers representing the number of hidden units in each layer.
    num_output (int): Number of output units in the final layer.
    prev_activations (list or None): List of activation functions used in previous layers, or None.

    Returns:
    tuple: Tuple containing lists of layers and activations.
    """

    layers = []
    activations = []
    for hidden in num_hidden:
        rng, inp_rng = jax.random.split(rng, 2)
        layers.append(nn.Dense(features=hidden))

        if prev_activations is None or len(num_hidden) != len(prev_activations):
            random_number = jax.random.uniform(inp_rng, shape=(1,)).item()
            if random_number < 0.25:
                activations.append(nn.relu)
            elif random_number < 0.5:
                activations.append(nn.sigmoid)
            elif random_number < 0.75:
                activations.append(nn.tanh)
            else:
                activations.append(nn.leaky_relu)
        else:
            activations = prev_activations

    layers.append(nn.Dense(features=num_output))

    return layers, activations


class GenomeClassifier(nn.Module):
    layers: list
    activations: list

    @nn.compact
    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        for layer_i in range(len(self.layers) - 1):
            x = self.layers[layer_i](x)
            x = self.activations[layer_i](x)

        x = self.layers[-1](x)

        return x


class SimpleClassifier(nn.Module):
    num_hidden: int  # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)

    def __call__(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x


def copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg):
    """
    Copy trained parameters over to a new model with potentially different layer structure.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_layers (int): Number of layers in the new model.
    trained_params (dict): Trained parameters from a previously trained model.
    prev_activations (list or None): List of activation functions used in previous layers, or None.
    cfg: Configuration object containing network parameters.

    Returns:
    tuple: Tuple containing new model and its initialized parameters.
    """

    layers, activations = create_layers(rng, num_layers, cfg.network.num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (cfg.network.num_inputs,))
    params = model.init(init_rng, inp)

    for layer, value in trained_params.items():
        if layer in params['params']:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(value['kernel']).copy().T

                # print(copy_params)
                # print(copy_train_params)
                # breakpoint()

                if len(copy_params) >= len(copy_train_params):
                    copy_params[:len(copy_train_params), :len(value['kernel'])] = copy_train_params[:,
                                                                                  :len(copy_params[0])]
                    params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)
                else:
                    copy_params = copy_train_params[:len(copy_params), :len(copy_params[0])]
                    params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)

            if 'bias' in value:
                copy_bias = jax.device_get(params['params'][layer]['bias']).copy()
                copy_train_bias = jax.device_get(value['bias']).copy()
                if len(copy_bias) > len(copy_train_bias):
                    copy_bias[:len(copy_train_bias)] = copy_train_bias
                    params['params'][layer]['bias'] = jax.numpy.array(copy_bias)
                else:
                    copy_bias[:len(copy_train_bias)] = copy_train_bias[:len(copy_bias)]
                    params['params'][layer]['bias'] = jax.numpy.array(copy_bias)

    return model, params


def add_new_layer(rng, num_layers, trained_params, cfg):
    num_layers.append(1)
    return copy_layers_over(rng, num_layers, trained_params, None, cfg)


def add_new_node(rng, num_layers, trained_params, prev_activations, cfg):
    """
    Add a new node to a randomly selected layer in the model.

    Parameters:
    rng (jax.random.PRNGKey): Random number generator key.
    num_layers (list): List containing the number of layers in the model.
    trained_params (dict): Trained parameters from a previously trained model.
    prev_activations (list or None): List of activation functions used in previous layers, or None.
    cfg: Configuration object containing network parameters.

    Returns:
    tuple: Tuple containing new model and its initialized parameters with an additional node.
    """
    if len(num_layers) <= 0:
        num_layers.insert(0, 1)
    else:
        rng, inp_rng = jax.random.split(rng, 2)
        random_element = jax.random.choice(inp_rng, jnp.asarray(num_layers)).item()

        index = jnp.where(jnp.array(num_layers) == random_element)[0][0]
        num_layers[index] = random_element + 1

    return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)


def remove_node_over(rng, num_layers, trained_params, prev_activations, cfg):
    layers, activations = create_layers(rng, num_layers, cfg.network.num_output, prev_activations)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (cfg.network.num_inputs,))
    params = model.init(init_rng, inp)

    for layer, value in params['params'].items():
        if layer in trained_params:
            if 'kernel' in value:
                copy_params = jax.device_get(params['params'][layer]['kernel']).copy().T
                copy_train_params = jax.device_get(trained_params[layer]['kernel']).copy().T

                copy_params = copy_train_params[:len(copy_params), :len(copy_params[0])]
                params['params'][layer]['kernel'] = jax.numpy.array(copy_params.T)

            if 'bias' in value:
                copy_arr = jax.device_get(value['bias']).copy()
                for num_node_i in range(len(value['bias'].T)):
                    copy_arr[num_node_i] = jax.device_get(trained_params[layer]['bias']).T[num_node_i]

                params['params'][layer]['bias'] = jax.numpy.array(copy_arr)

    return model, params


def remove_node(rng, num_layers, trained_params, prev_activations, cfg):
    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)
    selected_node = None
    for i in range(0, len(num_layers) - 1):
        if num_layers[i] > num_layers[i + 1]:
            selected_node = i
            break
    if selected_node is not None:
        num_layers[selected_node] -= 1
    else:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)

    print(num_layers)
    return remove_node_over(rng, num_layers, trained_params, prev_activations, cfg)


def remove_layer(rng, num_layers, trained_params, prev_activations, cfg):
    if len(num_layers) == 1:
        return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)
    rng, inp_rng = jax.random.split(rng, 2)
    random_layer = jax.random.choice(inp_rng, jnp.asarray(num_layers[:-1])).item()
    num_layers.remove(random_layer)
    print(num_layers)
    return copy_layers_over(rng, num_layers, trained_params, prev_activations, cfg)



# @hydra.main(version_base=None, config_path="../task", config_name="config_mnist")
@hydra.main(version_base=None, config_path="../task", config_name="config_iris")
def main(cfg):

    rng = jax.random.PRNGKey(cfg.jax.PRNGKey)

    num_layers = list(cfg.network.num_layers)
    num_inputs = cfg.network.num_inputs
    num_output = cfg.network.num_output
    batch_size = cfg.training.batch_size
    learning_rate = cfg.training.lr

    # Model
    layers, activations = create_layers(rng, num_layers, num_output, prev_activations=None)
    model = GenomeClassifier(layers=layers, activations=activations)

    rng, inp_rng, init_rng = jax.random.split(rng, 3)
    inp = jax.random.normal(inp_rng, (num_inputs,))

    params = model.init(init_rng, inp)


    optimizer = optax.sgd(learning_rate=learning_rate)

    #Dataset 
    if cfg.dataset.dataset_type == "digits":
        print("Loading Digits Dataset")
        sklearn_dataset = datasets.load_digits()
        n_samples = len(sklearn_dataset.images)
        data = sklearn_dataset.images.reshape((n_samples, -1))
        X_train, X_test, y_train, y_test = train_test_split(data, sklearn_dataset.target, test_size=0.3, shuffle=False)
        train_dataset = DigitsDataset(X_train, y_train)
        test_dataset = DigitsDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
        print("Done.")

    if cfg.dataset.dataset_type == "iris":
        print("Loading Iris Dataset")
        iris_dataset = datasets.load_iris()
        X = iris_dataset.data
        y = iris_dataset.target
        X = (X - X.mean()) / np.std(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        train_dataset = IrisDataset(X_train, y_train)
        test_dataset = IrisDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
        print("Done.")

    if cfg.dataset.dataset_type == "mnist":
        print("Loading MNIST Dataset")
        train = MNIST(root='train', train=True, transform=mnist_transform, download=True)
        test = MNIST(root='test', train=False, transform=mnist_transform, download=True)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=mnist_collate_fn)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, collate_fn=mnist_collate_fn)
        print("Done.")

    for generation in range(cfg.training.generations):

        model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
                            
        trained_model_state = train_model(
                                state=model_state, 
                                train_data_loader=train_loader, 
                                test_data_loader=test_loader, 
                                num_epochs=cfg.training.num_epochs, 
                                generation=generation,
                                num_output=num_output
                            )
        
        eval_model(trained_model_state,
                   test_loader,
                   epoch=cfg.training.num_epochs,
                   generation=generation,
                   num_output=num_output)


        checkpoints.save_checkpoint(ckpt_dir='/tmp/checkpoints',  # Folder to save checkpoint in
                                target=trained_model_state,  # What to save. To only save parameters, use model_state.params
                                step=100,  # Training step or other metric to save best model on
                                prefix=f'model_generation_{generation}_',  # Checkpoint file name prefix
                                overwrite=True   # Overwrite existing checkpoint files
                            )
        

        # Drawing graph
        if cfg.utils.draw_graph:
            draw_graph(trained_model_state, cfg)
            plt.savefig(f"results/graph_gen{generation}.png")
            plt.close()


        # Evolution
        rng, add_layer_rng, add_node_rng, rmv_layer_rng, rmv_node_rng = jax.random.split(rng, 5)
        trained_params = trained_model_state.params['params']
        modified = False

        # Add new layer
        if jax.random.uniform(add_layer_rng, shape=(1,)).item() < cfg.neat.add_layer:
            print("Adding new layer")
            model, trained_params = add_new_layer(rng, num_layers, trained_params, cfg)
            modified = True

        # Add new node
        if jax.random.uniform(add_node_rng, shape=(1,)).item() < cfg.neat.add_node:
            print("Adding new node")
            model, trained_params = add_new_node(rng, num_layers, trained_params, activations, cfg)
            modified = True

        # # Remove node
        if jax.random.uniform(rmv_layer_rng, shape=(1,)).item() < cfg.neat.remove_node: 
            print("Removing node")
            model, trained_params = remove_node(rng, num_layers, trained_params, activations, cfg)
            modified = True

        # #Remove layer
        if jax.random.uniform(rmv_node_rng, shape=(1,)).item() < cfg.neat.remove_layer:   
            print("Removing layer") 
            model, trained_params = remove_layer(rng, num_layers, trained_params, activations, cfg)
            modified = True


        if modified:
            params = trained_params
        else:
            model, params = copy_layers_over(rng, num_layers, trained_params, activations, cfg)


         

if __name__ == "__main__":
    main()