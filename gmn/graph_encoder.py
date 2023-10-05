import tensorflow as tf

from basic_layers.mlp import MLP

KL = tf.keras.layers


class GraphEncoder(KL.Layer):
    def __init__(self,
                 node_hidden_sizes=None,
                 edge_hidden_sizes=None,
                 name=None):
        super(GraphEncoder, self).__init__(name=name)

        self.node_hidden_sizes = node_hidden_sizes
        self.edge_hidden_sizes = edge_hidden_sizes

    def build(self, input_shape):
        if self.node_hidden_sizes:
            self.node_encoder = MLP(
                self.node_hidden_sizes, name="node_encoder")

        if self.edge_hidden_sizes:
            self.edge_encoder = MLP(
                self.edge_hidden_sizes, name="edge_encoder")

        KL.Layer.build(self, input_shape)

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
