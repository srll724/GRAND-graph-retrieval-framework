import tensorflow as tf

from basic_layers.mlp import MLP
from basic_layers.graph_conv import GraphConv

KL = tf.keras.layers


class GCN(KL.Layer):
    def __init__(self,
                 FLAGS,
                 output_sizes,
                 activation="relu",
                 use_bias=True,
                 dropout_rate=None,
                 activate_final=False,
                 graph_pooling=False,
                 pool_type="sum",
                 name=None):
        super(GCN, self).__init__(name=name)

        self.gcn_mode = FLAGS.gcn_mode
        assert self.gcn_mode == 2

        self.output_sizes = output_sizes
        self.activation = activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.activate_final = activate_final
        self.graph_pooling = graph_pooling
        self.pool_type = pool_type
        assert self.pool_type == "sum"
        assert self.graph_pooling

        self.layers = []
        for i, size in enumerate(self.output_sizes):
            if i < len(self.output_sizes) - 1 or self.activate_final:
                act = self.activation
            else:
                act = None
            self.layers.append(GraphConv(
                units=size,
                activation=act,
                use_bias=self.use_bias,
                dropout_rate=self.dropout_rate,
                name=f"GraphConv_{i}"))

        if self.graph_pooling:
            if self.gcn_mode == 2:
                ga_node_hidden_sizes = list(FLAGS.ga_node_hidden_sizes)
                ga_node_hidden_sizes[-1] = ga_node_hidden_sizes[-1] * 2
                self.node_transform_net = MLP(
                    output_sizes=ga_node_hidden_sizes,
                    name="node_transform_net")
                self.graph_transform_net = MLP(
                    output_sizes=FLAGS.graph_hidden_sizes,
                    name="graph_transform_net")

    def call(self, graph_data, training):
        results = []
        outputs = graph_data.node_features
        for layer in self.layers:
            outputs = layer(
                features=outputs,
                from_idx=graph_data.hatA_from_idx,
                to_idx=graph_data.hatA_to_idx,
                weights=graph_data.hatA_weights,
                training=training)
            results.append(outputs)
        if not self.graph_pooling:
            return outputs, results
        else:
            if self.gcn_mode == 1:
                assert False
                if self.pool_type == "sum":
                    pool_op = tf.math.unsorted_segment_sum
                else:
                    raise ValueError
                with tf.name_scope("graph_pooling"):
                    graph_idx, n_graphs = graph_data.graph_idx, graph_data.n_graphs
                    outputs = pool_op(outputs, graph_idx, n_graphs)
                    return outputs, results[-1]
            elif self.gcn_mode == 2:
                node_states = self.node_transform(outputs, training=training)
                # node_states = outputs
                graph_states = tf.math.unsorted_segment_sum(
                    node_states, graph_data.graph_idx, graph_data.n_graphs)
                graph_states = self.graph_transform_net(
                    graph_states, training=training)
                return graph_states, node_states

    def node_transform(self, node_states, training):
        node_states = self.node_transform_net(node_states, training=training)
        if True:
            s0, s1 = tf.split(
                node_states, num_or_size_splits=2, axis=1)
            gates = tf.nn.sigmoid(s0)
            node_states = gates * s1
        return node_states
