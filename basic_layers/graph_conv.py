import tensorflow as tf

KL = tf.keras.layers
KA = tf.keras.activations


class GraphConv(KL.Layer):
    def __init__(self,
                 units,
                 activation,
                 use_bias,
                 dropout_rate=None,
                 name=None):
        super(GraphConv, self).__init__(name=name)

        self.units = units
        self.activation = KA.get(activation)
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight(
            "kernel",
            shape=(last_dim, self.units),
            initializer="glorot_uniform",
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=(self.units,),
                initializer="zeros",
                dtype=self.dtype,
                trainable=True)

    def call(self, features, from_idx, to_idx, weights, training):
        use_dropout = self.dropout_rate not in (None, 0)

        if training and use_dropout:
            if isinstance(features, tf.sparse.SparseTensor):
                features = sparse_dropout(features, self.dropout_rate)
            else:
                features = tf.nn.dropout(features, self.dropout_rate)

        if isinstance(features, tf.sparse.SparseTensor):
            # sparse_dense_matmul introduces non-determinism on GPU
            # states = tf.sparse.sparse_dense_matmul(features, self.kernel)
            dense_features = tf.sparse.to_dense(features)
            states = tf.matmul(dense_features, self.kernel)
        else:
            states = tf.matmul(features, self.kernel)

        with tf.name_scope("to_states"):
            to_states = tf.gather(states, to_idx)
            weighted_to_states = weights[:, tf.newaxis] * to_states
        with tf.name_scope("state_aggregation"):
            outputs = tf.math.unsorted_segment_sum(
                weighted_to_states, from_idx, tf.shape(states)[0])

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs


def sparse_dropout(x, rate):
    assert isinstance(x, tf.sparse.SparseTensor)
    out = tf.sparse.SparseTensor(
        indices=x.indices, values=tf.nn.dropout(x.values, rate),
        dense_shape=x.dense_shape)
    return out


if __name__ == "__main__":
    import networkx as nx
    import numpy as np
    import scipy as sp
    from dataset import Dataset
    from utils.graph_utils import batch_graphs
    from utils.gcn_utils import preprocess_adj, sparse_to_tuple

    ds = Dataset("aids700", 32, 0.25, "bfs",
                 "onehot", "type", "regression", 123)
    # train_data = ds.get_data("training")
    # graph_data = train_data[0]
    graphs, labels = ds.get_pairs(32)
    graph_data = batch_graphs(graphs)
    graph_data = graph_data._replace(
        node_features=tf.sparse.reorder(
            tf.sparse.SparseTensor(*graph_data.node_features)))

    gc = GraphConv(128, "relu", True, None)

    o = gc(graph_data.node_features,
           graph_data.from_idx, graph_data.to_idx,
           graph_data.weights,
           training=True)

    # original graph_conv implementation
    graphs = [g for g_tuple in graphs for g in g_tuple]
    oo = []
    for g in graphs:
        adj = nx.to_numpy_array(
            g, nodelist=range(g.number_of_nodes()), weight="dummy")
        np.fill_diagonal(adj, 0)
        support_tuple = preprocess_adj(adj)
        support = tf.cast(
            tf.sparse.reorder(tf.sparse.SparseTensor(*support_tuple)),
            tf.float32)

        feat_dim = g.graph["node_feat_dim"]
        feat_idx = [g.nodes[n]["feat_idx"]
                    for n in range(g.number_of_nodes())]
        feat_mx = sp.sparse.eye(feat_dim).tocsr()[feat_idx].astype(np.float32)
        feat_tuple = sparse_to_tuple(feat_mx)
        feat = tf.sparse.reorder(tf.sparse.SparseTensor(*feat_tuple))

        this_o = tf.sparse.sparse_dense_matmul(feat, gc.kernel)
        this_o = tf.sparse.sparse_dense_matmul(support, this_o)
        this_o = tf.nn.bias_add(this_o, gc.bias)
        this_o = gc.activation(this_o)
        oo.append(this_o)
    oo = tf.concat(oo, axis=0)

    tf.debugging.assert_near(o, oo)
