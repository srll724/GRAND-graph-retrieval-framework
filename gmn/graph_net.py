import tensorflow as tf

from gmn.graph_prop_layer import GraphPropLayer
from gmn.graph_prop_matching_layer import GraphPropMatchingLayer


KL = tf.keras.layers


class GraphNet(KL.Layer):
    def __init__(self,
                 encoder,
                 aggregator,
                 node_state_dim,
                 n_prop_layers,
                 share_prop_params,
                 model_type,
                 FLAGS,
                 name=None):
        super(GraphNet, self).__init__(name=name)

        self.encoder = encoder
        self.aggregator = aggregator
        self.node_state_dim = node_state_dim
        self.n_prop_layers = n_prop_layers
        self.share_prop_params = share_prop_params
        self.model_type = model_type
        self.FLAGS = FLAGS

        if self.model_type == "embedding":
            self.layer_class = GraphPropLayer
        elif self.model_type == "matching":
            self.layer_class = GraphPropMatchingLayer

        self.prop_layers = []

    def build(self, input_shape):
        FLAGS = self.FLAGS
        if self.model_type == "embedding":
            name_prefix = "graph_prop"
            extra_kwargs = {}
        elif self.model_type == "matching":
            name_prefix = "graph_prop_matching"
            extra_kwargs = {
                "matching_similarity_type": FLAGS.matching_similarity_type}

        for i in range(self.n_prop_layers):
            if i == 0 or not self.share_prop_params:
                layer = self.layer_class(
                    node_state_dim=FLAGS.node_state_dim,
                    edge_hidden_sizes=FLAGS.gp_edge_hidden_sizes,
                    node_hidden_sizes=FLAGS.gp_node_hidden_sizes,
                    edge_net_init_scale=FLAGS.edge_net_init_scale,
                    node_update_type=FLAGS.node_update_type,
                    use_reverse_direction=FLAGS.use_reverse_direction,
                    reverse_param_different=FLAGS.reverse_param_different,
                    use_layer_norm=FLAGS.use_layer_norm,
                    name=f"{name_prefix}_{i}",
                    **extra_kwargs)
            else:
                layer = self.prop_layers[0]
            self.prop_layers.append(layer)

        KL.Layer.build(self, input_shape)

    def call(self, from_idx, to_idx,
             node_features, edge_features,
             graph_idx, n_graphs, training):
        raw_edge_features = edge_features
        node_features, edge_features = self.encoder(
            node_features, edge_features, training=training)

        node_states = node_features
        if self.model_type == "embedding":
            for layer in self.prop_layers:
                node_states = layer(
                    node_states, from_idx, to_idx, training=training,
                    node_features=None, edge_features=raw_edge_features)
        elif self.model_type == "matching":
            for layer in self.prop_layers:
                node_states = layer(
                    node_states, from_idx, to_idx,
                    graph_idx, n_graphs, training=training,
                    node_features=None, edge_features=raw_edge_features)

        outputs, node_states = self.aggregator(
            node_states, graph_idx, n_graphs, training=training)

        return outputs, node_states
