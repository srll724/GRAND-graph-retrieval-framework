import tensorflow as tf

from basic_layers import MLP, GCN, GNN
from graph_sim.pairwise_node_similarity import PairwiseNodeSimilarity
from graph_sim.separate_cnn import SeparateCNN
# from gmn.model import GMNModel

KL = tf.keras.layers


class GraphSimNet(KL.Layer):
    def __init__(self, FLAGS, trainable=True, name=None):
        super(GraphSimNet, self).__init__(trainable=trainable, name=name)

        self.FLAGS = FLAGS
        self.graphsim_encoder = FLAGS.graphsim_encoder
        # self.gcn = GCN(
        #     output_sizes=FLAGS.gcn_sizes,
        #     activation="relu",
        #     use_bias=True,
        #     dropout_rate=None,
        #     activate_final=False)
        # self.gnn = GMNModel(FLAGS, node_or_graph="node")
        if self.graphsim_encoder == "gnn":
            self.encoder = GNN(
                ge_node_hidden_sizes=FLAGS.ge_node_hidden_sizes,
                ge_edge_hidden_sizes=FLAGS.ge_edge_hidden_sizes,
                mp_edge_hidden_sizes=FLAGS.gp_edge_hidden_sizes,
                mp_node_hidden_sizes=(
                    [FLAGS.node_state_dim] if FLAGS.node_update_type == "gru"
                    else FLAGS.gp_node_hidden_sizes),
                edge_net_init_scale=FLAGS.edge_net_init_scale,
                node_update_type=FLAGS.node_update_type,
                use_reverse_direction=FLAGS.use_reverse_direction,
                reverse_param_different=FLAGS.reverse_param_different,
                use_layer_norm=FLAGS.use_layer_norm,
                n_mp_layers=FLAGS.n_prop_layers,
                share_mp_params=FLAGS.share_prop_params,
                ga_node_hidden_sizes=FLAGS.ga_node_hidden_sizes,
                gated=FLAGS.gated,
                graph_hidden_sizes=FLAGS.graph_hidden_sizes,
                node_or_graph="node")
        elif self.graphsim_encoder == "gcn":
            self.encoder = GCN(
                FLAGS=FLAGS,
                output_sizes=FLAGS.gcn_sizes,
                activation="relu",
                use_bias=True,
                dropout_rate=None,
                activate_final=False,
                graph_pooling=True,
                pool_type="sum")
        self.pns = PairwiseNodeSimilarity(fixed_size=FLAGS.fixed_size)
        self.n_splits = FLAGS.n_prop_layers if FLAGS.use_all_node_states else 1
        assert self.n_splits == 1
        self.cnn = SeparateCNN(
            n_splits=self.n_splits,
            filter_sizes=FLAGS.cnn_filter_sizes,
            kernel_sizes=FLAGS.cnn_kernel_sizes,
            pool_sizes=FLAGS.cnn_pool_sizes,
            padding="SAME",
            pool_type="max",
            activation="relu",
            use_bias=True,
            activate_final=True)
        self.mlp = MLP(
            output_sizes=FLAGS.mlp_sizes,
            w_init=None,
            b_init=None,
            use_bias=True,
            activation="relu",
            dropout_rate=None,
            activate_final=False)

    def call(self, graph_data, n_graphs, training):
        assert isinstance(n_graphs, int) and n_graphs % 2 == 0, n_graphs
        # _, node_states_list = self.gcn(graph_data, training=training)
        _, node_states_list = self.encoder(graph_data, training=training)
        ##########################
        if self.n_splits == 1 and self.graphsim_encoder == 'gnn':
            node_states_list = node_states_list[-1:]
        elif self.n_splits == 1 and self.graphsim_encoder == 'gcn':
            node_states_list = [node_states_list]
        ##########################
        resized_sim_list, sim_list = [], []
        for i in range(self.n_splits):
            this_sim = self.pns(
                node_states_list[i], graph_data.graph_idx, n_graphs)
            resized_sim_list.append(this_sim[0])
            sim_list.append(this_sim[1])
        sim = tf.stack(resized_sim_list, axis=-1)
        cnn_outputs = self.cnn(sim)
        outputs = self.mlp(cnn_outputs, training=training)
        return outputs, node_states_list[-1]


if __name__ == "__main__":
    from tensorflow.python.ops import summary_ops_v2
    from flags import FLAGS
    from dataset import Dataset
    from utils.graph_utils import GraphData

    ds = Dataset("aids700", "regression", "bfs",
                 "onehot", "type", 0.25, 32, 1000, 123)
    train_data = ds.get_data("training")
    graph_data = train_data[0]

    assert False

    model = GraphSimNet(FLAGS)

    n_graphs = ds.n_graphs_per_batch_train
    # have to initialize gcn weights in eager mode
    o = model(graph_data, n_graphs, training=True)

    def forward(graph_data):
        outputs = model(graph_data, n_graphs, training=True)
        return outputs

    tf_forward = tf.function(
        forward,
        input_signature=[
            GraphData(
                from_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
                to_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
                weights=tf.TensorSpec(shape=[None], dtype=tf.float32),
                node_features=tf.SparseTensorSpec(shape=[None, 29], dtype=tf.float32),
                edge_features=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
                graph_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
                n_graphs=tf.TensorSpec(shape=[], dtype=tf.int32))])
    # oo = tf_forward(graph_data)
    # ooo = tf_forward(graph_data)

    # tf.debugging.assert_near(o, oo)
    # tf.debugging.assert_equal(oo, ooo)

    writer = tf.summary.create_file_writer(logdir="./tb_visual/forward")
    with writer.as_default():
        summary_ops_v2.graph(tf_forward.get_concrete_function().graph)
