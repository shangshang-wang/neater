import jax
import jax.numpy as jnp
import optax
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from functools import partial
import random
import math
import time
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.datasets import make_gaussian_quantiles


default_config = {
    'pop_size': 50,
    'num_inputs': 2,
    'num_outputs': 1,
    'output_activation': 'sigmoid',
    'initial_connection': 'direct',
    'feed_forward': True, # If True, prevent recurrent connections

    'weight_init_mean': 0.0, 'weight_init_stdev': 1.0,
    'weight_min_value': -5.0, 'weight_max_value': 5.0,
    'bias_init_mean': 0.0, 'bias_init_stdev': 1.0,
    'bias_min_value': -5.0, 'bias_max_value': 5.0,
    'response_init_mean': 1.0, 'response_init_stdev': 0.1,
    'response_min_value': 0.1, 'response_max_value': 5.0,

    'node_add_prob': 0.05,
    'conn_add_prob': 0.1,
    'conn_delete_prob': 0.05,
    'node_delete_prob': 0.05,

    'bias_mutate_rate': 0.7,
    'bias_replace_rate': 0.1,
    'bias_mutate_power': 0.5,
    'response_mutate_rate': 0.7,
    'response_replace_rate': 0.1,
    'response_mutate_power': 0.5,
    'weight_mutate_rate': 0.8,
    'weight_replace_rate': 0.1,
    'weight_mutate_power': 0.5,
    'enabled_mutate_rate': 0.05,

    'activation_default': 'relu',
    'activation_options': ['relu', 'sigmoid', 'tanh', 'identity', 'sin', 'gauss', 'abs', 'square'],
    'activation_mutate_rate': 0.1,

    'compatibility_threshold': 3.0,
    'compatibility_disjoint_coefficient': 1.0,
    'compatibility_weight_coefficient': 0.5,

    'elitism': 2,
    'survival_threshold': 0.2,

    'bp_learning_rate': 0.01,
    'bp_batch_size': 32,
    'bp_epochs': 50,
    'complexity_penalty': 0.001
}

activation_functions = {
    'relu': jax.nn.relu,
    'sigmoid': jax.nn.sigmoid,
    'tanh': jnp.tanh,
    'identity': lambda x: x,
    'sin': jnp.sin,
    'cos': jnp.cos,
    'gauss': lambda x: jnp.exp(-jnp.multiply(x, x) / 2.0),
    'abs': jnp.abs,
    'square': jnp.square,
    'cube': lambda x: jnp.power(x, 3),
    'clamped': lambda x: jnp.clip(x, -1.0, 1.0),
    'log': lambda x: jnp.log(jnp.maximum(x, 1e-7)),
    'exp': lambda x: jnp.exp(jnp.clip(x, -60.0, 60.0)),
    'hat': lambda x: jnp.maximum(0.0, 1.0 - jnp.abs(x)),
}


class NodeGene:
    # Types: 0=input, 1=output, 2=hidden
    def __init__(self, id, type, bias=0.0, response=1.0, activation='relu'):
        self.id = id
        self.type = type
        self.bias = float(bias)
        self.response = float(response)
        self.activation = activation

    def __repr__(self):
        types = {0: 'In', 1: 'Out', 2: 'Hid'}
        return f"Node({self.id}, {types.get(self.type, 'Unk')}, act={self.activation}, b={self.bias:.2f}, r={self.response:.2f})"

    def copy(self):
        return NodeGene(self.id, self.type, self.bias, self.response, self.activation)

    def mutate(self, config):
        if random.random() < config.get('bias_replace_rate', 0.1):
            self.bias = random.gauss(config.get('bias_init_mean', 0.0), config.get('bias_init_stdev', 1.0))
        elif random.random() < config.get('bias_mutate_rate', 0.7):
            self.bias += random.gauss(0, config.get('bias_mutate_power', 0.5))
        self.bias = np.clip(self.bias, config.get('bias_min_value', -5.0), config.get('bias_max_value', 5.0))

        if random.random() < config.get('response_replace_rate', 0.1):
            self.response = random.gauss(config.get('response_init_mean', 1.0), config.get('response_init_stdev', 0.1))
        elif random.random() < config.get('response_mutate_rate', 0.7):
            self.response += random.gauss(0, config.get('response_mutate_power', 0.5))
        self.response = np.clip(self.response, config.get('response_min_value', 0.1), config.get('response_max_value', 5.0))

        if self.type == 2 and random.random() < config.get('activation_mutate_rate', 0.1):
             available_activations = config.get('activation_options', ['relu', 'sigmoid', 'tanh'])
             valid_options = [act for act in available_activations if act in activation_functions]
             if valid_options:
                 self.activation = random.choice(valid_options)


class ConnectionGene:
    def __init__(self, in_node, out_node, weight, enabled, innovation):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = float(weight)
        self.enabled = enabled
        self.innovation = innovation

    def __repr__(self):
        return f"Conn({self.in_node}->{self.out_node}, w={self.weight:.2f}, {'E' if self.enabled else 'D'}, I:{self.innovation})"

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation)

    def mutate(self, config):
        if random.random() < config.get('weight_replace_rate', 0.1):
            self.weight = random.gauss(config.get('weight_init_mean', 0.0), config.get('weight_init_stdev', 1.0))
        elif random.random() < config.get('weight_mutate_rate', 0.8):
             self.weight += random.gauss(0, config.get('weight_mutate_power', 0.5))
        self.weight = np.clip(self.weight, config.get('weight_min_value', -5.0), config.get('weight_max_value', 5.0))


class Genome:
    def __init__(self, key, num_inputs, num_outputs, config):
        self.key = key
        self.nodes = {}
        self.connections = {}
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness = -float('inf')
        self.node_id_counter = 0
        self.config = config
        self.global_innovation_number = 0
        self.innovation_history = {}

        input_node_ids = []
        for i in range(num_inputs):
            node_id = -(i + 1)
            bias = random.gauss(self.config.get('bias_init_mean', 0.0), self.config.get('bias_init_stdev', 1.0))
            response = random.gauss(self.config.get('response_init_mean', 1.0), self.config.get('response_init_stdev', 0.1))
            self.nodes[node_id] = NodeGene(node_id, type=0, bias=bias, response=response, activation='identity')
            self.node_id_counter = min(self.node_id_counter, node_id)
            input_node_ids.append(node_id)

        output_node_ids = []
        for i in range(num_outputs):
            node_id = i
            bias = random.gauss(self.config.get('bias_init_mean', 0.0), self.config.get('bias_init_stdev', 1.0))
            response = random.gauss(self.config.get('response_init_mean', 1.0), self.config.get('response_init_stdev', 0.1))
            out_act = self.config.get('output_activation', 'sigmoid')
            self.nodes[node_id] = NodeGene(node_id, type=1, bias=bias, response=response, activation=out_act)
            self.node_id_counter = max(self.node_id_counter, node_id)
            output_node_ids.append(node_id)

        self.node_id_counter = max(0, self.node_id_counter) + 1

        init_conn_type = self.config.get('initial_connection', 'direct')
        if init_conn_type == 'direct':
            for i_id in input_node_ids:
                for o_id in output_node_ids:
                    mean = self.config.get('weight_init_mean', 0.0)
                    stdev = self.config.get('weight_init_stdev', 1.0)
                    weight = random.gauss(mean, stdev)
                    weight = np.clip(weight, self.config.get('weight_min_value', -5.0), self.config.get('weight_max_value', 5.0))
                    self._add_connection(i_id, o_id, weight, enabled=True)


    def get_new_node_id(self):
        new_id = self.node_id_counter
        self.node_id_counter += 1
        return new_id

    def get_innovation_number(self, in_node_id, out_node_id):
        key = (in_node_id, out_node_id)
        if key in self.innovation_history:
            return self.innovation_history[key]
        else:
            self.global_innovation_number += 1
            self.innovation_history[key] = self.global_innovation_number
            return self.global_innovation_number

    def _add_node(self, node_id, type, bias=0.0, response=1.0, activation='relu'):
        if node_id not in self.nodes:
             self.nodes[node_id] = NodeGene(node_id, type, bias, response, activation)
        return self.nodes[node_id]

    def _add_connection(self, in_node_id, out_node_id, weight, enabled):
        key = (in_node_id, out_node_id)
        innovation_num = self.get_innovation_number(in_node_id, out_node_id)
        if key not in self.connections:
             self.connections[key] = ConnectionGene(in_node_id, out_node_id, weight, enabled, innovation_num)
        return self.connections[key]

    def mutate(self):
        if random.random() < self.config.get('node_add_prob', 0.05):
            self.mutate_add_node()

        if random.random() < self.config.get('conn_add_prob', 0.1):
            self.mutate_add_connection()

        for node in self.nodes.values():
            if node.type != 0:
                node.mutate(self.config)
        for conn in self.connections.values():
             conn.mutate(self.config)
             if random.random() < self.config.get('enabled_mutate_rate', 0.05):
                  conn.enabled = not conn.enabled

    def mutate_add_node(self):
        if not self.connections: return

        enabled_connections = {k: v for k, v in self.connections.items() if v.enabled}
        if not enabled_connections: return

        conn_key = random.choice(list(enabled_connections.keys()))
        in_node_id, out_node_id = conn_key
        old_conn = self.connections[conn_key]

        old_conn.enabled = False

        new_node_id = self.get_new_node_id()
        hidden_act = self.config.get('activation_default', 'relu')
        bias = random.gauss(self.config.get('bias_init_mean', 0.0), self.config.get('bias_init_stdev', 1.0))
        response = random.gauss(self.config.get('response_init_mean', 1.0), self.config.get('response_init_stdev', 0.1))
        self._add_node(new_node_id, type=2, bias=bias, response=response, activation=hidden_act)

        self._add_connection(in_node_id, new_node_id, 1.0, enabled=True)
        self._add_connection(new_node_id, out_node_id, old_conn.weight, enabled=True)

    def mutate_add_connection(self):
        possible_starts = [n for n in self.nodes.values()]
        possible_ends = [n for n in self.nodes.values() if n.type != 'input']

        if not possible_starts or not possible_ends: return

        is_feedforward = self.config.get('feed_forward', True)

        attempts = 20
        for _ in range(attempts):
            start_node = random.choice(possible_starts)
            end_node = random.choice(possible_ends)
            start_node_id = start_node.id
            end_node_id = end_node.id
            key_conn = (start_node_id, end_node_id)

            if key_conn in self.connections:
                continue
            if start_node_id == end_node_id:
                continue
            if end_node.type == 'input':
                continue

            if is_feedforward:
                 graph_check = nx.DiGraph()
                 graph_check.add_nodes_from(self.nodes.keys())
                 existing_enabled_edges = [(c.in_node, c.out_node) for c in self.connections.values() if c.enabled]
                 graph_check.add_edges_from(existing_enabled_edges)
                 if start_node_id in graph_check and end_node_id in graph_check:
                     graph_check.add_edge(start_node_id, end_node_id)
                 else:
                     print(f"Warning: Node missing during cycle check setup ({start_node_id} or {end_node_id})")
                     continue
                 if not nx.is_directed_acyclic_graph(graph_check):
                     continue

            mean = self.config.get('weight_init_mean', 0.0)
            stdev = self.config.get('weight_init_stdev', 1.0)
            weight = random.gauss(mean, stdev)
            weight = np.clip(weight, self.config.get('weight_min_value', -5.0), self.config.get('weight_max_value', 5.0))
            self._add_connection(start_node_id, end_node_id, weight, enabled=True)
            return

    @staticmethod
    def crossover(parent1, parent2, child_key, innovation_tracker):
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1

        config = parent1.config
        child = Genome(child_key, config['num_inputs'], config['num_outputs'], config)
        child.global_innovation_number = innovation_tracker['global_innovation_number']
        child.innovation_history = innovation_tracker['innovation_history']
        child.node_id_counter = max(parent1.node_id_counter, parent2.node_id_counter)

        child.nodes = {}
        child.connections = {}

        for node_id, node1 in parent1.nodes.items():
            child.nodes[node_id] = node1.copy()

        conn1_by_innov = {c.innovation: c for c in parent1.connections.values()}
        conn2_by_innov = {c.innovation: c for c in parent2.connections.values()}
        all_innovations = set(conn1_by_innov.keys()) | set(conn2_by_innov.keys())

        for innov in sorted(list(all_innovations)):
            gene1 = conn1_by_innov.get(innov)
            gene2 = conn2_by_innov.get(innov)

            chosen_conn_copy = None
            inherit_enabled = True

            if gene1 and gene2:
                chosen_gene = random.choice([gene1, gene2])
                chosen_conn_copy = chosen_gene.copy()
                if not gene1.enabled or not gene2.enabled:
                    inherit_enabled = (random.random() > 0.75)
            elif gene1:
                chosen_conn_copy = gene1.copy()
                inherit_enabled = gene1.enabled
            elif gene2:
                 chosen_conn_copy = gene2.copy()
                 inherit_enabled = gene2.enabled

            if chosen_conn_copy:
                if chosen_conn_copy.in_node not in child.nodes:
                    if chosen_conn_copy.in_node in parent2.nodes:
                        child.nodes[chosen_conn_copy.in_node] = parent2.nodes[chosen_conn_copy.in_node].copy()
                    else:
                        continue
                if chosen_conn_copy.out_node not in child.nodes:
                    if chosen_conn_copy.out_node in parent2.nodes:
                        child.nodes[chosen_conn_copy.out_node] = parent2.nodes[chosen_conn_copy.out_node].copy()
                    else:
                        continue

                chosen_conn_copy.enabled = inherit_enabled
                child.connections[(chosen_conn_copy.in_node, chosen_conn_copy.out_node)] = chosen_conn_copy

        innovation_tracker['global_innovation_number'] = child.global_innovation_number
        innovation_tracker['innovation_history'] = child.innovation_history

        return child


def genome_to_forward_fn(genome: Genome):
    nodes_list = list(genome.nodes.values())
    connections_list = [c for c in genome.connections.values() if c.enabled]

    initial_params = {
        'weights': {(c.in_node, c.out_node): c.weight for c in connections_list},
        'biases': {n.id: n.bias for n in nodes_list if n.type != 0},
        'responses': {n.id: n.response for n in nodes_list if n.type != 0}
    }

    graph = nx.DiGraph()
    node_ids = [n.id for n in nodes_list]
    graph.add_nodes_from(node_ids)
    edges = [(c.in_node, c.out_node) for c in connections_list]
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        if genome.config.get('feed_forward', True):
             print(f"Warning: Genome {genome.key} has cycles but config requires feedforward. Evaluation might fail.")

    try:
        eval_order = list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
         print(f"Warning: Genome {genome.key} graph may not be fully connected or has issues. Eval may fail.")
         input_nodes = [n.id for n in nodes_list if n.type == 0]
         output_nodes = [n.id for n in nodes_list if n.type == 1]
         hidden_nodes = [n.id for n in nodes_list if n.type == 2]
         eval_order = input_nodes + hidden_nodes + output_nodes

    input_node_ids = frozenset(n.id for n in nodes_list if n.type == 0)
    output_node_ids = frozenset(n.id for n in nodes_list if n.type == 1)
    node_activation_map = {n.id: activation_functions.get(n.activation, activation_functions['identity'])
                           for n in nodes_list}
    incoming_connections = defaultdict(list)
    for c in connections_list:
        incoming_connections[c.out_node].append(c.in_node)

    def forward_pass(params, inputs):
        if inputs.ndim == 1: inputs = inputs.reshape(1, -1)
        batch_size = inputs.shape[0]

        node_values = {node_id: jnp.zeros(batch_size) for node_id in node_ids}

        input_idx = 0
        for node_id in sorted(list(input_node_ids)):
            if input_idx < inputs.shape[1]:
                node_values[node_id] = inputs[:, input_idx]
                input_idx += 1

        for node_id in eval_order:
            if node_id in input_node_ids:
                continue

            bias = params['biases'].get(node_id, 0.0)
            response = params['responses'].get(node_id, 1.0)
            activation_fn = node_activation_map[node_id]

            node_sum = jnp.zeros(batch_size)
            for in_node_id in incoming_connections[node_id]:
                 weight = params['weights'].get((in_node_id, node_id), 0.0)
                 node_sum += node_values[in_node_id] * weight

            node_values[node_id] = activation_fn(bias + response * node_sum)

        outputs = jnp.stack([node_values[out_id] for out_id in sorted(list(output_node_ids))], axis=-1)
        return outputs.reshape(batch_size, len(output_node_ids))

    return forward_pass, initial_params


def binary_cross_entropy_loss(logits, labels):
    probs = jnp.clip(logits, 1e-7, 1.0 - 1e-7)
    return -jnp.mean(labels * jnp.log(probs) + (1 - labels) * jnp.log(1 - probs))

def calculate_accuracy(apply_fn, params, X, y):
    y_pred_prob = apply_fn(params, X)
    y_pred = (y_pred_prob > 0.5).astype(jnp.int32)
    return jnp.mean(y_pred == y)

@partial(jax.jit, static_argnums=(0, 3))
def train_step(apply_fn, params, opt_state, optimizer, X_batch, y_batch):
    def loss_fn_for_grad(p):
        y_pred_logits = apply_fn(p, X_batch)
        loss = binary_cross_entropy_loss(y_pred_logits, y_batch)
        return loss

    loss, grads = jax.value_and_grad(loss_fn_for_grad)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def evaluate_genome_fitness(key, genome: Genome, X, y):
    config = genome.config
    complexity_penalty_rate = config.get('complexity_penalty', 0.001)

    graph = nx.DiGraph()
    nodes_list = list(genome.nodes.keys())
    connections_list = [c for c in genome.connections.values() if c.enabled]
    edges = [(c.in_node, c.out_node) for c in connections_list]

    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        return 0.0

    apply_fn, initial_params = genome_to_forward_fn(genome)

    has_params = any(bool(p_dict) for p_dict in initial_params.values() if isinstance(p_dict, dict)) or any(initial_params.values())
    if not has_params:
         return 0.0

    key, train_key = jax.random.split(key)
    trained_params, final_loss, final_accuracy = train_network(
        train_key, apply_fn, initial_params, X, y, config
    )

    num_nodes = len(genome.nodes)
    num_connections = len(connections_list)
    penalty = complexity_penalty_rate * (num_nodes + num_connections)
    fitness = final_accuracy - penalty
    return float(max(0.0, fitness))


def train_network(key, apply_fn, initial_params, X_train, y_train, config):
    learning_rate = config.get('bp_learning_rate', 0.01)
    batch_size = config.get('bp_batch_size', 32)
    epochs = config.get('bp_epochs', 50)

    optimizer = optax.adam(learning_rate)
    params = initial_params

    if not any(bool(p_dict) for p_dict in params.values() if isinstance(p_dict, dict)) and not any(params.values()):
        print("Warning: Empty initial parameters, cannot train.")
        return params, float('inf'), 0.0

    try:
        opt_state = optimizer.init(params)
    except Exception as e:
         print(f"Error initializing optimizer state: {e}. Parameters: {params}")
         return initial_params, float('inf'), 0.0

    num_train = X_train.shape[0]
    steps_per_epoch = math.ceil(num_train / batch_size)

    for epoch in range(epochs):
        key, subkey = jax.random.split(key)
        perms = jax.random.permutation(subkey, num_train)
        X_train_perm, y_train_perm = X_train[perms], y_train[perms]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = start_idx + batch_size
            X_batch, y_batch = X_train_perm[start_idx:end_idx], y_train_perm[start_idx:end_idx]
            if X_batch.shape[0] == 0: continue

            params, opt_state, loss = train_step(apply_fn, params, opt_state, optimizer, X_batch, y_batch)
            epoch_loss += loss

    final_loss = binary_cross_entropy_loss(apply_fn(params, X_train), y_train)
    final_accuracy = calculate_accuracy(apply_fn, params, X_train, y_train)
    return params, final_loss, final_accuracy


def get_xor_data(key, n_samples=200):
    rng = np.random.RandomState(key.tolist()[0])
    X = rng.rand(n_samples, 2) * 2 - 1
    y_xor = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    y = y_xor.astype(int)
    return jnp.array(X), jnp.array(y).reshape(-1, 1)

def get_circles_data(key, n_samples=200, noise=0.1, factor=0.5):
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=key.tolist()[0])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return jnp.array(X), jnp.array(y).reshape(-1, 1)

def get_spiral_data(key, n_samples=200, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=key.tolist()[0])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return jnp.array(X), jnp.array(y).reshape(-1, 1)

def get_checkerboard_data(key, n_samples=400, grid_size=4, noise_std=0.1):
    rng = np.random.RandomState(key.tolist()[0])

    grid_range = 2.0
    square_size = (2 * grid_range) / grid_size

    X = rng.uniform(-grid_range, grid_range, size=(n_samples, 2))

    x_indices = np.floor((X[:, 0] + grid_range) / square_size)
    y_indices = np.floor((X[:, 1] + grid_range) / square_size)
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    y = ((x_indices + y_indices) % 2).astype(int)
    X += rng.normal(scale=noise_std, size=X.shape)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32).reshape(-1, 1)


def get_gaussian_quantiles_data(key, n_samples=400, n_classes=2, noise_std=0.5):
    X, y = make_gaussian_quantiles(mean=None,
                                   cov=noise_std**2,
                                   n_samples=n_samples,
                                   n_features=2,
                                   n_classes=n_classes,
                                   shuffle=True,
                                   random_state=key.tolist()[0])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32).reshape(-1, 1)

def get_interlocking_spirals_data(key, n_samples=400, noise=0.2, turns=1.5):
    rng = np.random.RandomState(key.tolist()[0])
    n = n_samples // 2

    theta0 = np.sqrt(rng.rand(n)) * turns * (2*np.pi)
    r0 = theta0 / np.pi * 0.5
    x0 = (r0 * np.cos(theta0) + rng.randn(n) * noise).reshape(n, 1)
    y0 = (r0 * np.sin(theta0) + rng.randn(n) * noise).reshape(n, 1)

    theta1 = np.sqrt(rng.rand(n)) * turns * (2*np.pi)
    r1 = theta1 / np.pi * 0.5
    x1 = (-r1 * np.cos(theta1) + rng.randn(n) * noise).reshape(n, 1)
    y1 = (-r1 * np.sin(theta1) + rng.randn(n) * noise).reshape(n, 1)

    X = np.vstack((np.hstack((x0, y0)), np.hstack((x1, y1))))
    y = np.hstack((np.zeros(n), np.ones(n)))

    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.float32).reshape(-1, 1)


def plot_decision_boundary(apply_fn, params, X, y, title="Decision Boundary"):
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    try:
        Z = apply_fn(params, grid_points)
        if Z.shape[-1] == 1: Z = (Z > 0.5).astype(int)
        else: Z = jnp.argmax(Z, axis=-1)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
        plt.xlabel("Feature 1 (X)")
        plt.ylabel("Feature 2 (Y)")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
        plt.savefig(f"decision_boundary_backprop.png")
    except Exception as e:
        print(f"Error during decision boundary plotting: {e}")
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
        plt.title(f"{title} (Boundary Plot Failed)")

def visualize_network(genome: Genome, title="Evolved Network Structure"):
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()
    node_labels = {}
    node_colors = []
    edge_labels = {}

    pos = {}
    input_nodes = sorted([n.id for n in genome.nodes.values() if n.type == 0])
    output_nodes = sorted([n.id for n in genome.nodes.values() if n.type == 1])
    hidden_nodes = sorted([n.id for n in genome.nodes.values() if n.type == 2])

    node_subsets = {0: input_nodes, 1: hidden_nodes, 2: output_nodes}
    subset_map = {node_id: 0 for node_id in input_nodes}
    subset_map.update({node_id: 1 for node_id in hidden_nodes})
    subset_map.update({node_id: 2 for node_id in output_nodes})

    max_nodes_in_layer = max(len(input_nodes), len(hidden_nodes), len(output_nodes), 1)
    node_yspacing = 1.0 / max(1, max_nodes_in_layer -1) if max_nodes_in_layer > 1 else 0
    layer_x = {0: 0.0, 1: 0.5, 2: 1.0}

    layer_counters = defaultdict(int)
    for node_id in input_nodes + hidden_nodes + output_nodes:
        node = genome.nodes[node_id]
        G.add_node(node_id)
        node_labels[node_id] = f"{node_id}\n{node.activation[:3]}\nb={node.bias:.1f} r={node.response:.1f}"

        layer = subset_map[node_id]
        y_pos = 0.5 - layer_counters[layer] * node_yspacing * (max_nodes_in_layer / (len(node_subsets[layer]) if len(node_subsets[layer])>0 else 1))
        pos[node_id] = (layer_x[layer], y_pos)
        layer_counters[layer] += 1

        if node.type == 0: node_colors.append('lightblue')
        elif node.type == 1: node_colors.append('lightcoral')
        else: node_colors.append('lightgray')

    edge_list = []
    for conn_key, conn in genome.connections.items():
        if conn.enabled:
            G.add_edge(conn.in_node, conn.out_node, weight=conn.weight)
            edge_labels[conn_key] = f"{conn.weight:.2f}"

    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color=node_colors,
            node_size=2500, font_size=8, font_weight='bold', arrowsize=15, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(title)
    plt.axis('off')
    plt.savefig(f"network_backprop.png")


def main(key, config, task_name='XOR', generations=100):
    innovation_tracker = {
        'global_innovation_number': 0,
        'innovation_history': {}
    }

    key, data_key = jax.random.split(key)
    if task_name == 'XOR': X, y = get_xor_data(data_key)
    elif task_name == 'Circles': X, y = get_circles_data(data_key)
    elif task_name == 'Spiral': X, y = get_spiral_data(data_key)
    elif task_name == 'Checkerboard': X, y = get_checkerboard_data(data_key)
    elif task_name == 'GaussianQuantiles': X, y = get_gaussian_quantiles_data(data_key)
    elif task_name == 'InterlockingSpirals': X, y = get_interlocking_spirals_data(data_key)
    else: raise ValueError(f"Unknown task: {task_name}")

    num_inputs = config['num_inputs']
    num_outputs = config['num_outputs']
    population_size = config['pop_size']
    elitism = config.get('elitism', 2)
    survival_threshold = config.get('survival_threshold', 0.2)

    print(f"Running BackpropNEAT for {task_name}...")
    print(f"Population: {population_size}, Generations: {generations}")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")

    population = []
    keys = jax.random.split(key, population_size)
    for i in range(population_size):
        g = Genome(i, num_inputs, num_outputs, config)
        if i == 0:
             innovation_tracker['global_innovation_number'] = g.global_innovation_number
             innovation_tracker['innovation_history'] = g.innovation_history
        else:
             g.global_innovation_number = innovation_tracker['global_innovation_number']
             g.innovation_history = innovation_tracker['innovation_history']
        population.append(g)

    best_genome_overall = None
    best_fitness_overall = -float('inf')

    for gen in range(generations):
        key, eval_key, repro_key = jax.random.split(key, 3)

        fitness_scores = []
        evaluation_keys = jax.random.split(eval_key, population_size)
        for i, genome in enumerate(population):
            genome.fitness = evaluate_genome_fitness(evaluation_keys[i], genome, X, y)
            fitness_scores.append(genome.fitness)
            innovation_tracker['global_innovation_number'] = max(innovation_tracker['global_innovation_number'], genome.global_innovation_number)
            innovation_tracker['innovation_history'].update(genome.innovation_history)

        population.sort(key=lambda g: g.fitness, reverse=True)

        best_gen_genome = population[0]
        best_gen_fitness = best_gen_genome.fitness
        avg_gen_fitness = np.mean([g.fitness for g in population if g.fitness > -float('inf')])

        if best_gen_fitness > best_fitness_overall:
            best_fitness_overall = best_gen_fitness
            best_genome_overall = copy.deepcopy(best_gen_genome)
            print(f"*** Gen {gen+1}: New best! Fitness: {best_fitness_overall:.4f} (Avg: {avg_gen_fitness:.4f}) ***")
        else:
             print(f"Gen {gen+1}: Best Fitness={best_gen_fitness:.4f} (Avg: {avg_gen_fitness:.4f}) (Overall Best: {best_fitness_overall:.4f})")

        next_population = []

        for i in range(min(elitism, population_size)):
             next_population.append(population[i])

        num_offspring = population_size - len(next_population)
        parent_pool = population[:max(elitism, int(population_size * 0.5))]

        child_keys = jax.random.split(repro_key, num_offspring)
        child_id_start = population_size

        for i in range(num_offspring):
            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)

            # Crossover
            child_key = child_id_start + i
            child = Genome.crossover(parent1, parent2, child_key, innovation_tracker)

            # Mutation
            child.mutate()
            innovation_tracker['global_innovation_number'] = child.global_innovation_number
            innovation_tracker['innovation_history'] = child.innovation_history

            next_population.append(child)

        population = next_population

    print("\nEvolution Finished!")
    if best_genome_overall:
        print(f"Best Genome found with Fitness: {best_fitness_overall:.4f}")
    else:
        print("No suitable genome found.")

    return best_genome_overall, X, y


if __name__ == "__main__":
    master_key = jax.random.PRNGKey(int(time.time()))
    task_name = 'InterlockingSpirals' # XOR, Circles, Spiral, Checkerboard, GaussianQuantiles, InterlockingSpirals
    generations = 50  # Number of generations
    config = default_config.copy()
    config['pop_size'] = 100

    print(f"\n--- Running Task: {task_name} ---")
    key, task_key, retrain_key = jax.random.split(master_key, 3)

    best_genome, X, y = main(task_key, config=config, task_name=task_name, generations=generations)

    if best_genome:
        print(f"\n--- Visualizing Best Genome for {task_name} ---")

        visualize_network(best_genome, title=f"Best Evolved Network for {task_name}")
        plt.show()

        apply_fn, initial_params = genome_to_forward_fn(best_genome)

        if not initial_params.get('weights') and not initial_params.get('biases'):
            print("Best genome has no trainable parameters, cannot visualize boundary.")
        else:
            print("Retraining best genome for visualization...")
            retrain_config = config.copy()
            retrain_config['bp_epochs'] = 300
            trained_params, final_loss, final_accuracy = train_network(
                retrain_key, apply_fn, initial_params, X, y, retrain_config
            )
            print(f"Retrained Accuracy for {task_name}: {final_accuracy:.4f}")
            plot_decision_boundary(apply_fn, trained_params, X, y,
                                   title=f"{task_name} Classification\nFinal Accuracy: {final_accuracy:.2%}")
            plt.show()
