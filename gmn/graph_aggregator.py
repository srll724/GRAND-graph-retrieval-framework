import tensorflow as tf

from basic_layers.mlp import MLP

KL = tf.keras.layers


AGGREGATION_TYPE = {
    "sum": tf.math.unsorted_segment_sum,
    "mean": tf.math.unsorted_segment_mean,
    "sqrt_n": tf.math.unsorted_segment_sqrt_n,
    "max": tf.math.unsorted_segment_max}


class GraphAggregator(KL.Layer):
    def __init__(self,
                 node_hidden_sizes,
                 graph_hidden_sizes,
                 gated,
                 aggregation_type,
                 node_or_graph,
                 name=None):
        super(GraphAggregator, self).__init__(name=name)

        self.node_hidden_sizes = list(node_hidden_sizes)
        if gated:
            self.node_hidden_sizes[-1] = node_hidden_sizes[-1] * 2
        self.graph_hidden_sizes = graph_hidden_sizes
        # self.graph_state_dim = node_hidden_sizes[-1]
        self.gated = gated
        self.aggregation_type = aggregation_type
        self.aggregation_fn = AGGREGATION_TYPE[aggregation_type]
        self.node_or_graph = node_or_graph

    def build(self, input_shape):
        self.node_g_net = MLP(
            self.node_hidden_sizes, name="node_g_net")

        if self.graph_hidden_sizes:
            self.graph_net = MLP(
                self.graph_hidden_sizes, name="graph_net")

        KL.Layer.build(self, input_shape)

    def call(self, node_states, graph_idx, n_graphs, training):
        node_states_g = self.node_g_net(node_states, training=training)
        if self.gated:
            s0, s1 = tf.split(
                node_states_g, num_or_size_splits=2, axis=1)
            gates = tf.nn.sigmoid(s0)
            node_states_g = s1 * gates

        if self.node_or_graph == "node":
            return node_states_g
        elif self.node_or_graph == "graph":
            graph_states = self.aggregation_fn(node_states_g, graph_idx, n_graphs)

            if self.aggregation_type == "max":
                raise NotImplementedError

            if self.graph_hidden_sizes:
                graph_states = self.graph_net(graph_states, training=training)

            return graph_states, node_states_g
