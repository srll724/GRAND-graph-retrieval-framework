import tensorflow as tf

from gmn.graph_encoder import GraphEncoder
from gmn.graph_aggregator import GraphAggregator
from gmn.graph_net import GraphNet

KL = tf.keras.layers


class GMNModel(KL.Layer):
    def __init__(self, FLAGS, node_or_graph, trainable=True, name=None):
        super(GMNModel, self).__init__(trainable=trainable, name=name)

        self.FLAGS = FLAGS
        self.node_or_graph = node_or_graph

    def build(self, input_shape):
        FLAGS = self.FLAGS
        self.encoder = GraphEncoder(
            node_hidden_sizes=FLAGS.ge_node_hidden_sizes,
            edge_hidden_sizes=FLAGS.ge_edge_hidden_sizes)
        self.aggregator = GraphAggregator(
            node_hidden_sizes=FLAGS.ga_node_hidden_sizes,
            graph_hidden_sizes=FLAGS.graph_hidden_sizes,
            gated=FLAGS.gated,
            aggregation_type=FLAGS.aggregation_type,
            node_or_graph=self.node_or_graph)
        self.net = GraphNet(
            encoder=self.encoder,
            aggregator=self.aggregator,
            node_state_dim=FLAGS.node_state_dim,
            n_prop_layers=FLAGS.n_prop_layers,
            share_prop_params=FLAGS.share_prop_params,
            model_type="matching",
            FLAGS=FLAGS)

        KL.Layer.build(self, input_shape)

    def call(self, inputs, n_graphs, training):
        # import ipdb; ipdb.set_trace()
        # node_features = tf.sparse.to_dense(inputs.node_features)
        net_inputs = {
            "from_idx": inputs.from_idx,
            "to_idx": inputs.to_idx,
            "node_features": inputs.node_features,
            "edge_features": inputs.edge_features,
            "graph_idx": inputs.graph_idx,
            "n_graphs": n_graphs,
            "training": training}

        net_outputs = self.net(**net_inputs)

        return net_outputs
