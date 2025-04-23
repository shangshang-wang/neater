import jax
import jax.numpy as jnp
from jax import jit, vmap, lax # Import lax for switch
from functools import partial # For JIT compilation with static args

import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
import time # Added for seeding
from collections import defaultdict # For incoming_connections map
import numpy as np # Keep numpy for plotting and some non-JAX random choices
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler


# Adjusted for pure NEAT (no backprop params, higher mutation focus)
default_config = {
    'pop_size': 150, # Might need larger population for pure NEAT
    'num_inputs': 2,
    'num_outputs': 1,
    'output_activation': 'sigmoid',
    'initial_connection': 'direct',
    'feed_forward': True, # *** CONTROLS CYCLE CHECK ***

    # Attribute Initialization
    'weight_init_mean': 0.0, 'weight_init_stdev': 1.0,
    'weight_min_value': -5.0, 'weight_max_value': 5.0,
    'bias_init_mean': 0.0, 'bias_init_stdev': 1.0,
    'bias_min_value': -5.0, 'bias_max_value': 5.0,
    'response_init_mean': 1.0, 'response_init_stdev': 0.1,
    'response_min_value': 0.1, 'response_max_value': 5.0,

    # Mutation Probabilities
    'node_add_prob': 0.05,
    'conn_add_prob': 0.1,
    'conn_delete_prob': 0.05, # Not implemented yet
    'node_delete_prob': 0.05, # Not implemented yet

    # Attribute Mutation Rates & Powers (Increased rates as primary weight opt)
    'bias_mutate_rate': 0.8, # Increased
    'bias_replace_rate': 0.1,
    'bias_mutate_power': 0.5,
    'response_mutate_rate': 0.3,
    'response_replace_rate': 0.05,
    'response_mutate_power': 0.3,
    'weight_mutate_rate': 0.8, # Increased significantly
    'weight_replace_rate': 0.1,
    'weight_mutate_power': 0.5,
    'enabled_mutate_rate': 0.05,

    # Activation Function Mutation
    'activation_default': 'relu',
    'activation_options': ['relu', 'sigmoid', 'tanh', 'identity', 'sin', 'gauss', 'abs', 'square'],
    'activation_mutate_rate': 0.1,

    # Complexity Penalty
    'complexity_penalty': 0.001
}

# --- Activation Functions ---
# Define an ORDERED TUPLE of functions and a name-to-index map
activation_fn_list = (
    jax.nn.relu, jax.nn.sigmoid, jnp.tanh, lambda x: x, # identity
    jnp.sin, jnp.cos, lambda x: jnp.exp(-jnp.multiply(x, x) / 2.0), # gauss
    jnp.abs, jnp.square, lambda x: jnp.power(x, 3), # cube
    lambda x: jnp.clip(x, -1.0, 1.0), # clamped
    lambda x: jnp.log(jnp.maximum(x, 1e-7)), # log
    lambda x: jnp.exp(jnp.clip(x, -60.0, 60.0)), # exp
    lambda x: jnp.maximum(0.0, 1.0 - jnp.abs(x)) # hat
)
# Ensure names here match 'activation_options' in config and order matches list above
activation_name_to_index = {
    'relu': 0, 'sigmoid': 1, 'tanh': 2, 'identity': 3, 'sin': 4, 'cos': 5,
    'gauss': 6, 'abs': 7, 'square': 8, 'cube': 9, 'clamped': 10, 'log': 11,
    'exp': 12, 'hat': 13
}
# Default activation if name not found
default_activation_index = activation_name_to_index.get('identity', 3)


# --- NEAT Gene Classes ---
class NodeGene:
    def __init__(self, id, type, bias=0.0, response=1.0, activation='relu'):
        self.id = id
        self.type = type
        self.bias = float(bias)
        self.response = float(response)
        self.activation = activation # Keep name here

    def __repr__(self):
        types = {0: 'In', 1: 'Out', 2: 'Hid'}
        type_str = types.get(self.type, 'Unk')
        return f"Node({self.id}, {type_str}, act={self.activation}, b={self.bias:.2f}, r={self.response:.2f})"

    def copy(self):
        return NodeGene(self.id, self.type, self.bias, self.response, self.activation)

    def mutate(self, config):
        # Mutate Bias
        if random.random() < config.get('bias_replace_rate', 0.1):
            mean = config.get('bias_init_mean', 0.0)
            stdev = config.get('bias_init_stdev', 1.0)
            self.bias = random.gauss(mean, stdev)
        elif random.random() < config.get('bias_mutate_rate', 0.7):
            power = config.get('bias_mutate_power', 0.5)
            self.bias += random.gauss(0, power)

        min_val = config.get('bias_min_value', -5.0)
        max_val = config.get('bias_max_value', 5.0)
        self.bias = np.clip(self.bias, min_val, max_val) # Use numpy clip for scalar float

        # Mutate Response
        if random.random() < config.get('response_replace_rate', 0.1):
            mean = config.get('response_init_mean', 1.0)
            stdev = config.get('response_init_stdev', 0.1)
            self.response = random.gauss(mean, stdev)
        elif random.random() < config.get('response_mutate_rate', 0.7):
            power = config.get('response_mutate_power', 0.5)
            self.response += random.gauss(0, power)

        min_val = config.get('response_min_value', 0.1)
        max_val = config.get('response_max_value', 5.0)
        self.response = np.clip(self.response, min_val, max_val)

        # Mutate Activation (only for hidden nodes)
        if self.type == 2 and random.random() < config.get('activation_mutate_rate', 0.1):
             options = config.get('activation_options', ['relu'])
             # Use activation_name_to_index to ensure valid options
             valid_options = [name for name in options if name in activation_name_to_index]
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
        status = 'E' if self.enabled else 'D'
        return f"Conn({self.innovation}: {self.in_node}->{self.out_node}, w={self.weight:.2f}, {status})"

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.enabled, self.innovation)

    def mutate(self, config):
        # Mutate Weight
        if random.random() < config.get('weight_replace_rate', 0.1):
            mean = config.get('weight_init_mean', 0.0)
            stdev = config.get('weight_init_stdev', 1.0)
            self.weight = random.gauss(mean, stdev)
        elif random.random() < config.get('weight_mutate_rate', 0.8):
            power = config.get('weight_mutate_power', 0.5)
            self.weight += random.gauss(0, power)

        min_val = config.get('weight_min_value', -5.0)
        max_val = config.get('weight_max_value', 5.0)
        self.weight = np.clip(self.weight, min_val, max_val)


class Genome:
    def __init__(self, key, num_inputs, num_outputs, config):
        self.key = key # Unique identifier or JAX key info
        self.nodes = {} # Dict: node_id -> NodeGene
        self.connections = {} # Dict: (in_node_id, out_node_id) -> ConnectionGene
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.fitness = -float('inf')
        self.node_id_counter = 0 # Simple counter for new node IDs
        self.config = config # Store parameters

        # Innovation tracking (should ideally be global)
        self.global_innovation_number = 0
        self.innovation_history = {} # Key: (in_node_id, out_node_id), Value: innovation_number

        # --- Initial Structure ---
        input_node_ids = []
        node_id_counter_min = 0
        for i in range(num_inputs):
            node_id = -(i + 1)
            bias = random.gauss(self.config.get('bias_init_mean', 0.0), self.config.get('bias_init_stdev', 1.0))
            response = random.gauss(self.config.get('response_init_mean', 1.0), self.config.get('response_init_stdev', 0.1))
            self.nodes[node_id] = NodeGene(node_id, type=0, bias=bias, response=response, activation='identity')
            node_id_counter_min = min(node_id_counter_min, node_id)
            input_node_ids.append(node_id)

        output_node_ids = []
        node_id_counter_max = 0
        for i in range(num_outputs):
            node_id = i
            bias = random.gauss(self.config.get('bias_init_mean', 0.0), self.config.get('bias_init_stdev', 1.0))
            response = random.gauss(self.config.get('response_init_mean', 1.0), self.config.get('response_init_stdev', 0.1))
            out_act = self.config.get('output_activation', 'sigmoid')
            self.nodes[node_id] = NodeGene(node_id, type=1, bias=bias, response=response, activation=out_act)
            node_id_counter_max = max(node_id_counter_max, node_id)
            output_node_ids.append(node_id)

        # Set the next available node ID
        self.node_id_counter = node_id_counter_max + 1

        # Initial connectivity
        init_conn_type = self.config.get('initial_connection', 'direct')
        if init_conn_type == 'direct':
            for i_id in input_node_ids:
                for o_id in output_node_ids:
                    mean = self.config.get('weight_init_mean', 0.0)
                    stdev = self.config.get('weight_init_stdev', 1.0)
                    weight = random.gauss(mean, stdev)
                    min_val = self.config.get('weight_min_value', -5.0)
                    max_val = self.config.get('weight_max_value', 5.0)
                    weight = np.clip(weight, min_val, max_val)
                    self._add_connection(i_id, o_id, weight, enabled=True)

    def get_new_node_id(self):
        """Gets the next available node ID."""
        new_id = self.node_id_counter
        self.node_id_counter += 1
        return new_id

    def get_innovation_number(self, in_node_id, out_node_id):
        """Gets or creates an innovation number for a connection."""
        key = (in_node_id, out_node_id)
        if key in self.innovation_history:
            return self.innovation_history[key]
        else:
            self.global_innovation_number += 1
            self.innovation_history[key] = self.global_innovation_number
            return self.global_innovation_number

    def _add_node(self, node_id, type, bias=0.0, response=1.0, activation='relu'):
        """Helper to add node if it doesn't exist."""
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeGene(node_id, type, bias, response, activation)
        return self.nodes[node_id]

    def _add_connection(self, in_node_id, out_node_id, weight, enabled):
        """Helper to add connection if it doesn't exist."""
        key = (in_node_id, out_node_id)
        innovation_num = self.get_innovation_number(in_node_id, out_node_id)
        if key not in self.connections:
            self.connections[key] = ConnectionGene(in_node_id, out_node_id, weight, enabled, innovation_num)
        return self.connections[key]

    def mutate(self):
        """Applies various mutations based on probabilities in config."""
        if random.random() < self.config.get('node_add_prob', 0.05):
            self.mutate_add_node()

        if random.random() < self.config.get('conn_add_prob', 0.1):
            self.mutate_add_connection()

        # --- Attribute Mutations ---
        for node in self.nodes.values():
            if node.type != 0: # Don't mutate input node attributes
                node.mutate(self.config)
        for conn in self.connections.values():
             conn.mutate(self.config)
             # Mutate enabled status
             if random.random() < self.config.get('enabled_mutate_rate', 0.05):
                  conn.enabled = not conn.enabled

    def mutate_add_node(self):
        """Splits an existing connection by adding a new node."""
        if not self.connections:
            return
        enabled_connections = {k: v for k, v in self.connections.items() if v.enabled}
        if not enabled_connections:
            return
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

    def mutate_add_connection(self): # Uses the robust cycle check
        """Adds a new connection, attempting to avoid cycles if feed_forward=True."""
        possible_starts = [n for n in self.nodes.values()]
        possible_ends = [n for n in self.nodes.values() if n.type != 'input']

        if not possible_starts or not possible_ends:
            return

        is_feedforward = self.config.get('feed_forward', True)
        attempts = 20 # Define attempts variable

        for _ in range(attempts): # Now 'attempts' is defined
            start_node = random.choice(possible_starts)
            end_node = random.choice(possible_ends)
            start_node_id = start_node.id
            end_node_id = end_node.id
            key_conn = (start_node_id, end_node_id)

            # Basic Validity Checks
            if key_conn in self.connections:
                continue
            if start_node_id == end_node_id:
                continue
            if end_node.type == 'input':
                continue

            # Cycle Check
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

            # Add Connection
            mean = self.config.get('weight_init_mean', 0.0)
            stdev = self.config.get('weight_init_stdev', 1.0)
            weight = random.gauss(mean, stdev)
            min_val = self.config.get('weight_min_value', -5.0)
            max_val = self.config.get('weight_max_value', 5.0)
            weight = np.clip(weight, min_val, max_val)
            self._add_connection(start_node_id, end_node_id, weight, enabled=True)
            return # Success

    @staticmethod
    def crossover(parent1, parent2, child_key, innovation_tracker):
        """Performs crossover between two parent genomes."""
        if parent1.fitness < parent2.fitness:
            parent1, parent2 = parent2, parent1

        config = parent1.config
        child = Genome(child_key, config['num_inputs'], config['num_outputs'], config)
        # Pass global innovation state - important!
        child.global_innovation_number = innovation_tracker['global_innovation_number']
        child.innovation_history = innovation_tracker['innovation_history']
        child.node_id_counter = max(parent1.node_id_counter, parent2.node_id_counter)
        child.nodes = {}
        child.connections = {}

        # Inherit nodes from fitter parent
        for node_id, node1 in parent1.nodes.items():
            child.nodes[node_id] = node1.copy()

        # Inherit connections
        # Use connection key (in_node, out_node) for easier lookup? No, stick to innovation number.
        # Need to access connection dict by value.innovation key
        conn1_by_innov = {c.innovation: c for c in parent1.connections.values()}
        conn2_by_innov = {c.innovation: c for c in parent2.connections.values()}
        all_innovations = set(conn1_by_innov.keys()) | set(conn2_by_innov.keys())

        for innov in sorted(list(all_innovations)):
            gene1 = conn1_by_innov.get(innov)
            gene2 = conn2_by_innov.get(innov)
            chosen_conn_copy = None
            inherit_enabled = True

            if gene1 and gene2: # Matching
                chosen_gene = random.choice([gene1, gene2])
                chosen_conn_copy = chosen_gene.copy()
                if not gene1.enabled or not gene2.enabled:
                    inherit_enabled = (random.random() > 0.75) # Standard NEAT rule
            elif gene1: # Disjoint/Excess from parent1
                chosen_conn_copy = gene1.copy()
                inherit_enabled = gene1.enabled
            elif gene2: # Disjoint/Excess from parent2
                 chosen_conn_copy = gene2.copy()
                 inherit_enabled = gene2.enabled

            if chosen_conn_copy:
                 # Ensure nodes exist in child
                 if chosen_conn_copy.in_node not in child.nodes:
                      if chosen_conn_copy.in_node in parent2.nodes:
                           # Copy node from less fit parent if needed for connection
                           child.nodes[chosen_conn_copy.in_node] = parent2.nodes[chosen_conn_copy.in_node].copy()
                      else: continue # Skip connection if node truly missing
                 if chosen_conn_copy.out_node not in child.nodes:
                      if chosen_conn_copy.out_node in parent2.nodes:
                           child.nodes[chosen_conn_copy.out_node] = parent2.nodes[chosen_conn_copy.out_node].copy()
                      else: continue

                 chosen_conn_copy.enabled = inherit_enabled
                 # Use tuple key for connections dict
                 child.connections[(chosen_conn_copy.in_node, chosen_conn_copy.out_node)] = chosen_conn_copy

        # Update the global innovation state from child
        innovation_tracker['global_innovation_number'] = child.global_innovation_number
        innovation_tracker['innovation_history'] = child.innovation_history

        return child


# --- Genome to JAX Function Conversion ---
def genome_to_forward_fn(genome: Genome):
    """Creates a JAX-compatible forward pass function using lax.switch."""
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
    eval_order = []
    is_dag = False

    try:
        is_dag = nx.is_directed_acyclic_graph(graph)
        if is_dag:
            eval_order = list(nx.topological_sort(graph))
        else:
             if genome.config.get('feed_forward', True):
                 # Pass genome key or some identifier if available
                 print(f"Warning: Genome graph {genome.key} has cycles. Using basic order.")
             # Fallback order
             input_nodes = sorted([n.id for n in nodes_list if n.type == 0])
             hidden_nodes = sorted([n.id for n in nodes_list if n.type == 2])
             output_nodes = sorted([n.id for n in nodes_list if n.type == 1])
             eval_order = input_nodes + hidden_nodes + output_nodes
    except nx.NetworkXUnfeasible:
         print(f"Warning: Graph issue (disconnected?) {genome.key}. Using basic order.")
         input_nodes = sorted([n.id for n in nodes_list if n.type == 0])
         hidden_nodes = sorted([n.id for n in nodes_list if n.type == 2])
         output_nodes = sorted([n.id for n in nodes_list if n.type == 1])
         eval_order = input_nodes + hidden_nodes + output_nodes

    input_node_ids_set = frozenset(n.id for n in nodes_list if n.type == 0)
    output_node_ids_list = sorted([n.id for n in nodes_list if n.type == 1])
    num_output_nodes = len(output_node_ids_list)

    # Create map from node_id to activation function INDEX
    node_activation_indices = {
        n.id: activation_name_to_index.get(n.activation, default_activation_index)
        for n in nodes_list
    }
    # Convert index map to tuple of tuples for static arg
    node_activation_indices_static = tuple(sorted(node_activation_indices.items()))

    incoming_connections = defaultdict(list)
    for c in connections_list:
        incoming_connections[c.out_node].append(c.in_node) # Store only input node id
    incoming_connections_static = frozenset((k, tuple(v)) for k,v in incoming_connections.items())

    # Static arguments list
    static_argnames = [
        'eval_order_static', 'input_node_ids_static', 'output_node_ids_static',
        'incoming_connections_static', 'node_activation_indices_static',
        'activation_funcs_static', 'num_output_nodes_static'
    ]

    @partial(jit, static_argnames=static_argnames)
    def _apply_fn(params_dynamic, single_input, eval_order_static, input_node_ids_static, output_node_ids_static,
                 incoming_connections_static, node_activation_indices_static,
                 activation_funcs_static, num_output_nodes_static):
        """Internal JIT'd function for single forward pass using lax.switch."""
        # Ensure input has at least 1 dimension for indexing later
        if hasattr(single_input, 'shape') and not single_input.shape:
             single_input = single_input.reshape(1)

        # Reconstruct lookup dicts from static tuples inside JIT
        node_activation_index_dict = dict(node_activation_indices_static)
        incoming_connections_dict = dict(incoming_connections_static)

        node_values = {}
        input_idx = 0
        # Ensure consistent input node order
        for node_id in sorted(list(input_node_ids_static)):
             # Check index bounds
             if input_idx < single_input.shape[0]:
                  node_values[node_id] = single_input[input_idx]
                  input_idx += 1
             # else: # Handle case where genome has more input nodes than data features
             #     node_values[node_id] = 0.0 # Or some default value

        for node_id in eval_order_static:
            if node_id in input_node_ids_static:
                continue

            bias = params_dynamic['biases'].get(node_id, 0.0)
            response = params_dynamic['responses'].get(node_id, 1.0)
            # Get activation index for this node
            activation_index = node_activation_index_dict.get(node_id, default_activation_index)
            node_sum = jnp.array(0.0, dtype=jnp.float32)

            # Use reconstructed dict here
            if node_id in incoming_connections_dict:
                for in_node_id in incoming_connections_dict[node_id]:
                     # Weight key uses in_node, out_node from connection gene
                     weight = params_dynamic['weights'].get((in_node_id, node_id), 0.0)
                     # Default to 0.0 if input node value not calculated yet (can happen with basic order)
                     in_val = node_values.get(in_node_id, 0.0)
                     node_sum += in_val * weight

            # Apply activation using lax.switch
            operand = bias + response * node_sum
            # Ensure index is within bounds for the branches tuple
            safe_index = jnp.clip(activation_index, 0, len(activation_funcs_static)-1).astype(int)
            node_values[node_id] = lax.switch(safe_index, activation_funcs_static, operand)

        # Collect outputs in defined order
        outputs = jnp.stack([node_values.get(out_id, 0.0) for out_id in output_node_ids_static])
        final_shape = (num_output_nodes_static,)
        return outputs.reshape(final_shape)

    # Wrapper to handle static arguments for JIT
    def forward_pass_callable(params_param, inputs_param):
         # Pass the activation function list/tuple itself as a static arg
         return _apply_fn(params_param, inputs_param,
                          eval_order_static=tuple(eval_order),
                          input_node_ids_static=frozenset(input_node_ids_set),
                          output_node_ids_static=tuple(output_node_ids_list),
                          incoming_connections_static=incoming_connections_static,
                          node_activation_indices_static=node_activation_indices_static,
                          activation_funcs_static=activation_fn_list, # Pass tuple of functions
                          num_output_nodes_static=num_output_nodes)

    return forward_pass_callable, initial_params


# --- Fitness Evaluation ---
def evaluate_genome_fitness(key, genome: Genome, X, y):
    """Evaluates fitness using JAX forward pass WITHOUT backprop training.
       Includes explicit check for cycles and penalizes them."""
    config = genome.config
    complexity_penalty_rate = config.get('complexity_penalty', 0.001)
    num_samples = X.shape[0]

    try:
        # --- Check for Cycles Explicitly During Evaluation ---
        graph = nx.DiGraph()
        nodes_list = list(genome.nodes.keys()) # Get all node IDs
        connections_list = [c for c in genome.connections.values() if c.enabled] # Get enabled connections
        edges = [(c.in_node, c.out_node) for c in connections_list]

        graph.add_nodes_from(nodes_list) # Add all nodes that exist in the genome
        graph.add_edges_from(edges) # Add only the enabled edges

        # If the final enabled graph is NOT acyclic, penalize heavily
        if not nx.is_directed_acyclic_graph(graph):
            # print(f"Genome {genome.key} is cyclic! Assigning zero fitness.") # Optional Debug
            return 0.0 # Assign very low/zero fitness to cyclic genomes
        # --- End Cycle Check ---

        # --- Proceed with evaluation ONLY if acyclic ---
        apply_fn, current_params = genome_to_forward_fn(genome) # Function creation still needed

        # Check for params (weights usually indicate connections)
        if not current_params.get('weights') and not any(c.enabled for c in genome.connections.values()):
            return 0.0 # No enabled connections found earlier

        # Vmap the forward pass function
        batch_apply_fn = vmap(apply_fn, in_axes=(None, 0))
        predictions = batch_apply_fn(current_params, X)

        # Calculate Accuracy
        pred_binary = (predictions > 0.5).astype(jnp.int32)
        correct_count = jnp.sum(pred_binary == y.astype(jnp.int32))
        accuracy = correct_count / num_samples

        # Complexity Penalty
        num_nodes = len(genome.nodes)
        num_connections = len(connections_list) # Use pre-filtered list
        penalty = complexity_penalty_rate * (num_nodes + num_connections)

        # Fitness: Accuracy squared minus penalty
        fitness = accuracy**2 - penalty

        return float(max(0.0, fitness))

    except Exception as e:
        # Catch errors during function creation or execution as well
        import traceback
        print(f"Error evaluating genome {genome.key}: {e}\n{traceback.format_exc()}")
        return 0.0 # Penalize errors heavily


# --- Dataset Generation ---
def get_xor_data(key, n_samples=4):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return jnp.array(X), jnp.array(y)

def get_circles_data(key, n_samples=200, noise=0.1, factor=0.5):
    X_np, y_np = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=key.tolist()[0])
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    return jnp.array(X_np, dtype=jnp.float32), jnp.array(y_np, dtype=jnp.float32).reshape(-1, 1)

def get_spiral_data(key, n_samples=200, noise=0.1):
    X_np, y_np = make_moons(n_samples=n_samples, noise=noise, random_state=key.tolist()[0])
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_np)
    return jnp.array(X_np, dtype=jnp.float32), jnp.array(y_np, dtype=jnp.float32).reshape(-1, 1)


# --- Visualization Functions ---
def visualize_network(genome: Genome, title="Evolved Network Structure"):
    """Draws the network graph using NetworkX and Matplotlib."""
    G = nx.DiGraph()
    pos = {}
    node_labels = {}
    node_colors = []
    edge_labels = {}
    edge_colors = []
    edge_widths = []

    input_ids = sorted([n.id for n in genome.nodes.values() if n.type == 0])
    output_ids = sorted([n.id for n in genome.nodes.values() if n.type == 1])
    hidden_ids = sorted([n.id for n in genome.nodes.values() if n.type == 2])

    layer_gap = 2.0 # Increase horizontal spacing
    node_gap = 1.0
    max_nodes_in_layer = max(len(input_ids) if input_ids else 1,
                           len(output_ids) if output_ids else 1,
                           len(hidden_ids) if hidden_ids else 1)

    def get_y_pos(index, total_nodes):
        """Calculates centered vertical position."""
        if total_nodes <= 1:
            return 0.0
        return (total_nodes - 1) * node_gap / 2.0 - index * node_gap

    # Assign positions and node properties
    for i, node_id in enumerate(input_ids):
        pos[node_id] = (0 * layer_gap, get_y_pos(i, len(input_ids)))
        node_labels[node_id] = f"I{node_id}\n{genome.nodes[node_id].activation[:3]}"
        node_colors.append('lightblue')
        G.add_node(node_id)

    for i, node_id in enumerate(hidden_ids):
        pos[node_id] = (1 * layer_gap, get_y_pos(i, len(hidden_ids)))
        node_labels[node_id] = f"H{node_id}\n{genome.nodes[node_id].activation[:3]}\nb={genome.nodes[node_id].bias:.1f}\nr={genome.nodes[node_id].response:.1f}"
        node_colors.append('lightgreen')
        G.add_node(node_id)

    for i, node_id in enumerate(output_ids):
        pos[node_id] = (2 * layer_gap, get_y_pos(i, len(output_ids)))
        node_labels[node_id] = f"O{node_id}\n{genome.nodes[node_id].activation[:3]}\nb={genome.nodes[node_id].bias:.1f}\nr={genome.nodes[node_id].response:.1f}"
        node_colors.append('lightcoral')
        G.add_node(node_id)

    # Add edges
    for conn in genome.connections.values():
        if conn.enabled:
            # Check if nodes exist in pos dict before adding edge
            if conn.in_node in pos and conn.out_node in pos:
                G.add_edge(conn.in_node, conn.out_node)
                edge_labels[(conn.in_node, conn.out_node)] = f"{conn.weight:.2f}"
                # Color edge based on weight sign, width based on magnitude
                color = 'red' if conn.weight < 0 else 'blue'
                width = max(0.1, min(3.0, abs(conn.weight))) # Clamp width
                edge_colors.append(color)
                edge_widths.append(width)
            else:
                # This might happen if nodes were added but layout calc failed?
                print(f"Warning: Node position missing for edge {conn.in_node}->{conn.out_node}")

    plt.figure(figsize=(12, 7))
    # Draw edges first
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, arrowsize=15)
    # Draw nodes on top
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors)
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_color='black')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='darkgrey')

    plt.title(title)
    plt.axis('off')
    plt.show()


def visualize_xor_performance(genome, title="XOR Performance"):
    """Plots the network's output vs target for XOR inputs using JAX activation."""
    xor_inputs_jnp, xor_outputs_jnp = get_xor_data(None)
    targets = xor_outputs_jnp.flatten()
    actuals = jnp.full_like(targets, 0.5) # Default

    try:
        apply_fn, current_params = genome_to_forward_fn(genome)
        if current_params.get('weights') or current_params.get('biases'):
            batch_apply_fn = vmap(apply_fn, in_axes=(None, 0))
            predictions = batch_apply_fn(current_params, xor_inputs_jnp)
            actuals = predictions.flatten()
    except Exception as e:
        print(f"Error activating network for XOR viz: {e}")

    # Convert JAX arrays to numpy for plotting
    targets_np = np.array(targets)
    actuals_np = np.array(actuals)
    inputs_np = np.array(xor_inputs_jnp)

    x = np.arange(len(inputs_np))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, targets_np, width, label='Target', color='orange')
    rects2 = ax.bar(x + width/2, actuals_np, width, label='Actual Output', color='steelblue')
    ax.set_ylabel('Output Value')
    ax.set_xlabel('XOR Input Pattern')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(i[0])},{int(i[1])}" for i in inputs_np])
    ax.set_ylim(-0.1, 1.1)
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_decision_boundary(genome, X, y, title="Decision Boundary"):
    """Plots decision boundary using the genome's current state."""
    plt.figure(figsize=(8, 6))
    X_np, y_np = np.array(X), np.array(y) # Ensure numpy for plotting calculations
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    h = 0.03
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points_np = np.c_[xx.ravel(), yy.ravel()]
    grid_points_jnp = jnp.array(grid_points_np, dtype=jnp.float32)
    Z = np.full(xx.shape, 0.5) # Default background

    try:
        apply_fn, current_params = genome_to_forward_fn(genome)
        if current_params.get('weights') or current_params.get('biases'):
            batch_apply_fn = vmap(apply_fn, in_axes=(None, 0))
            Z_jnp = batch_apply_fn(current_params, grid_points_jnp)
            if Z_jnp.shape[-1] == 1:
                Z_jnp = (Z_jnp > 0.5).astype(int)
            else: # Handle cases with >1 output node if necessary
                Z_jnp = jnp.argmax(Z_jnp, axis=-1)
            Z = np.array(Z_jnp.reshape(xx.shape))

        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
        plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
    except Exception as e:
        import traceback
        print(f"Error plotting boundary: {e}\n{traceback.format_exc()}")
        plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np.flatten(), cmap=plt.cm.RdBu, edgecolors='k')
        plt.title(f"{title} (Boundary Plot Failed)")
    plt.show()


# --- Main Evolutionary Loop ---
def run_neat_evolution(key, config, task_name='XOR', generations=100):
    """Main Pure NEAT loop (no backprop)."""
    # Centralized innovation tracking
    innovation_tracker = {'global_innovation_number': 0, 'innovation_history': {}}
    key, data_key = jax.random.split(key)

    if task_name == 'XOR':
        X, y = get_xor_data(data_key)
    elif task_name == 'Circles':
        X, y = get_circles_data(data_key)
        config['num_inputs']=2
        config['num_outputs']=1
    elif task_name == 'Spiral':
        X, y = get_spiral_data(data_key)
        config['num_inputs']=2
        config['num_outputs']=1
    else:
        raise ValueError(f"Unknown task: {task_name}")

    num_inputs = config['num_inputs']
    num_outputs = config['num_outputs']
    population_size = config['pop_size']
    elitism = config.get('elitism', 2)

    print(f"Running Pure NEAT (JAX Forward Pass) for {task_name}...")
    print(f"Population: {population_size}, Generations: {generations}")

    # Initialize Population
    population = []
    keys = jax.random.split(key, population_size)
    key = keys[0] # Reuse the first key for subsequent splits in the loop
    for i in range(population_size):
        # Pass index i as the genome's key for identification
        g = Genome(i, num_inputs, num_outputs, config)
        # Initialize global tracker from first genome
        if i == 0:
            innovation_tracker['global_innovation_number'] = g.global_innovation_number
            innovation_tracker['innovation_history'] = g.innovation_history
        else: # Share tracker state
            g.global_innovation_number = innovation_tracker['global_innovation_number']
            g.innovation_history = innovation_tracker['innovation_history']
        population.append(g)

    best_genome_overall = None
    best_fitness_overall = -float('inf')
    start_time = time.time()

    # --- Evolution Loop ---
    for gen in range(generations):
        key, eval_key, repro_key = jax.random.split(key, 3)
        eval_keys = jax.random.split(eval_key, population_size)

        # Evaluate Fitness
        fitness_scores = []
        for i, genome in enumerate(population):
            genome.fitness = evaluate_genome_fitness(eval_keys[i], genome, X, y)
            fitness_scores.append(genome.fitness)
            # Update global innovation state from genome
            innovation_tracker['global_innovation_number'] = max(innovation_tracker['global_innovation_number'], genome.global_innovation_number)
            innovation_tracker['innovation_history'].update(genome.innovation_history)

        # Sort population
        population.sort(key=lambda g: g.fitness, reverse=True)
        best_gen_genome = population[0]
        best_gen_fitness = best_gen_genome.fitness
        valid_fitness = [f for f in fitness_scores if f > -float('inf')]
        avg_gen_fitness = np.mean(valid_fitness) if valid_fitness else -float('inf')

        # Reporting and Storing Best
        if best_gen_fitness > best_fitness_overall:
            best_fitness_overall = best_gen_fitness
            best_genome_overall = copy.deepcopy(best_gen_genome)
            print(f"*** Gen {gen+1}: New best! Fitness: {best_fitness_overall:.4f} (Avg: {avg_gen_fitness:.4f}) ***")
        elif (gen + 1) % 10 == 0: # Print progress every 10 generations
             print(f"Gen {gen+1}: Best Fitness={best_gen_fitness:.4f} (Avg: {avg_gen_fitness:.4f}) (Overall: {best_fitness_overall:.4f})")

        # --- Reproduction ---
        next_population = []
        # Elitism
        for i in range(min(elitism, population_size)):
             # Elites are directly copied to next generation
             next_population.append(population[i])

        # Generate Offspring
        num_offspring = population_size - len(next_population)
        parent_pool = population[:max(elitism, int(population_size * 0.5))]
        # Assign new keys/IDs to children for tracking if needed
        child_id_start = population_size + gen * population_size

        for i in range(num_offspring):
            parent1 = random.choice(parent_pool)
            parent2 = random.choice(parent_pool)
            if parent1 == parent2 and len(parent_pool) > 1:
                 # Ensure different parents if possible
                 parent2 = random.choice([p for p in parent_pool if p != parent1] + [parent_pool[0]])

            child_key = child_id_start + i
            # Crossover passes the innovation tracker
            child = Genome.crossover(parent1, parent2, child_key, innovation_tracker)
            child.mutate() # Mutate child
            # Update global tracker state from child
            innovation_tracker['global_innovation_number'] = child.global_innovation_number
            innovation_tracker['innovation_history'] = child.innovation_history
            next_population.append(child)

        population = next_population

    # --- End of Evolution ---
    total_time = time.time() - start_time
    print(f"\nEvolution Finished in {total_time:.2f} seconds!")
    if best_genome_overall:
        print(f"Best Genome found with Fitness: {best_fitness_overall:.4f}")
    else:
        print("No suitable genome found.")
    return best_genome_overall, X, y


# --- Main Execution ---
if __name__ == "__main__":
    print("Using JAX backend:", jax.default_backend())
    print("Available JAX devices:", jax.devices())
    master_key = jax.random.PRNGKey(int(time.time())) # Seed from time

    # --- Task Setup ---
    task = 'XOR' # Choose 'XOR', 'Circles', or 'Spiral'
    config = default_config.copy() # Start with defaults
    config['pop_size'] = 200      # Population size
    generations = 300             # Number of generations (might need more for pure NEAT)

    # Task-specific adjustments
    if task == 'XOR':
        config['complexity_penalty'] = 0.005 # Penalize complexity more?
        config['num_inputs'] = 2
        config['num_outputs'] = 1
    elif task == 'Circles' or task == 'Spiral':
        config['complexity_penalty'] = 0.001 # Lower penalty for potentially complex tasks
        config['node_add_prob'] = 0.07       # Slightly higher mutation rates
        config['conn_add_prob'] = 0.12
        config['num_inputs'] = 2
        config['num_outputs'] = 1

    print(f"\n--- Running Task: {task} ---")
    print(f"Config: Pop={config['pop_size']}, Gens={generations}, Complexity Penalty={config['complexity_penalty']:.4f}")
    key, task_key = jax.random.split(master_key)

    # Run evolution
    best_genome, X, y = run_neat_evolution(task_key, config=config, task_name=task, generations=generations)

    # Visualize results
    if best_genome:
        print(f"\n--- Visualizing Best Genome for {task} ---")
        visualize_network(best_genome, title=f"Best Evolved Network for {task}")
        if task == 'XOR':
             visualize_xor_performance(best_genome, title=f"XOR Performance (Best Genome)")
        # Plot decision boundary for 2D tasks (XOR, Circles, Spiral)
        plot_decision_boundary(best_genome, X, y, title=f"{task} Classification (Best Genome)")

    print("\n--- Script Finished ---")