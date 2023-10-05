import tensorflow as tf

from basic_layers.mlp import MLP
from basic_layers.message_passing import MessagePassing

KL = tf.keras.layers


class GNN(KL.Layer):
    def __init__(self,
                 ge_node_hidden_sizes,
                 ge_edge_hidden_sizes,
                 mp_edge_hidden_sizes,
                 mp_node_hidden_sizes,
                 edge_net_init_scale,
                 node_update_type,
                 use_reverse_direction,
                 reverse_param_different,
                 use_layer_norm,
                 n_mp_layers,
                 share_mp_params,
                 ga_node_hidden_sizes,
                 gated,
                 graph_hidden_sizes,
                 node_or_graph,
                 name=None):
        assert node_or_graph in ["node", "graph"]

        super(GNN, self).__init__(name=name)

        # graph encoder
        self.ge_node_hidden_sizes = ge_node_hidden_sizes
        self.ge_edge_hidden_sizes = ge_edge_hidden_sizes
        # message passing
        self.mp_edge_hidden_sizes = mp_edge_hidden_sizes
        self.mp_node_hidden_sizes = mp_node_hidden_sizes
        self.edge_net_init_scale = edge_net_init_scale
        self.node_update_type = node_update_type
        self.use_reverse_direction = use_reverse_direction
        self.reverse_param_different = reverse_param_different
        self.use_layer_norm = use_layer_norm
        self.n_mp_layers = n_mp_layers
        self.share_mp_params = share_mp_params
        # graph aggregator
        ga_node_hidden_sizes = list(ga_node_hidden_sizes)
        ga_node_hidden_sizes[-1] = ga_node_hidden_sizes[-1] * 2
        self.ga_node_hidden_sizes = ga_node_hidden_sizes
        self.gated = gated
        self.graph_hidden_sizes = graph_hidden_sizes
        self.node_or_graph = node_or_graph

        self.graph_encoder = GraphEncoder(
            ge_node_hidden_sizes,
            ge_edge_hidden_sizes,
            name="graph_encoder")
        self.mp_layers = []
        for i in range(n_mp_layers):
            if i == 0 or not share_mp_params:
                layer = MessagePassing(
                    mp_edge_hidden_sizes,
                    mp_node_hidden_sizes,
                    edge_net_init_scale,
                    node_update_type,
                    use_reverse_direction,
                    reverse_param_different,
                    use_layer_norm,
                    name=f"message_passing_{i}")
            else:
                layer = self.mp_layers[0]
            self.mp_layers.append(layer)
        self.node_transform_net = MLP(
            output_sizes=ga_node_hidden_sizes,
            name="node_transform_net")
        if graph_hidden_sizes:
            self.graph_transform_net = MLP(
                output_sizes=graph_hidden_sizes,
                name="graph_transform_net")

    def call(self, graph_data, training):
        node_features, edge_features = self.graph_encoder(
            graph_data.node_features,
            graph_data.edge_features,
            training=training)

        node_states_list = []
        node_states = node_features
        for layer in self.mp_layers:
            node_states = layer(
                node_states, graph_data.from_idx, graph_data.to_idx,
                training=training, node_features=None,
                edge_features=graph_data.edge_features)
            node_states_list.append(node_states)
        transformed_node_states_list = [
            self.node_transform(state, training=training)
            for state in node_states_list]
        node_states = transformed_node_states_list[-1]

        if self.node_or_graph == "node":
            return node_states, transformed_node_states_list
        elif self.node_or_graph == "graph":
            graph_states = tf.math.unsorted_segment_sum(
                node_states, graph_data.graph_idx, graph_data.n_graphs)
            if self.graph_hidden_sizes:
                graph_states = self.graph_transform_net(
                    graph_states, training=training)
            return graph_states, node_states

    def node_transform(self, node_states, training):
        node_states = self.node_transform_net(node_states, training=training)
        if self.gated:
            s0, s1 = tf.split(
                node_states, num_or_size_splits=2, axis=1)
            gates = tf.nn.sigmoid(s0)
            node_states = gates * s1
        return node_states


class GraphEncoder(KL.Layer):
    def __init__(self,
                 node_hidden_sizes,
                 edge_hidden_sizes,
                 name=None):
        super(GraphEncoder, self).__init__(name=name)

        self.node_hidden_sizes = node_hidden_sizes
        self.edge_hidden_sizes = edge_hidden_sizes

        if node_hidden_sizes:
            self.node_encoder = MLP(
                output_sizes=node_hidden_sizes,
                name="node_encoder")
        else:
            self.node_encoder = None
        if edge_hidden_sizes:
            self.edge_encoder = MLP(
                output_sizes=edge_hidden_sizes,
                name="edge_encoder")
        else:
            self.edge_encoder = None

    def call(self, node_features, edge_features, training):
        if self.node_hidden_sizes:
            node_outputs = self.node_encoder(node_features, training=training)
        else:
            node_outputs = node_features
        if self.edge_hidden_sizes:
            edge_outputs = self.edge_encoder(edge_features, training=training)
        else:
            edge_outputs = edge_features
        return node_outputs, edge_outputs
