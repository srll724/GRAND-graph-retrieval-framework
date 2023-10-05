import tensorflow as tf

from graph_sim.graph_sim_net import GraphSimNet
from basic_layers import GCN, GNN
from gmn.model import GMNModel
from kd_loss import KDLoss

KL = tf.keras.layers


class Model(KL.Layer):
    def __init__(self, FLAGS, name=None):
        super(Model, self).__init__(name=name)

        self.FLAGS = FLAGS
        self.model_name = FLAGS.model_name

        if self.model_name == "graphsim":
            self.model = GraphSimNet(FLAGS)
        elif self.model_name == "gcn":
            self.model = GCN(
                FLAGS=FLAGS,
                output_sizes=FLAGS.gcn_sizes,
                activation="relu",
                use_bias=True,
                dropout_rate=None,
                activate_final=False,
                graph_pooling=True,
                pool_type="sum")
        elif self.model_name == "gnn":
            self.model = GNN(
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
                node_or_graph="graph")
        elif self.model_name == "gmn":
            self.model = GMNModel(FLAGS, node_or_graph="graph")
        elif self.model_name in ["kd", "dml"]:
            if FLAGS.student_name == "gcn":
                self.student = GCN(
                    FLAGS=FLAGS,
                    output_sizes=FLAGS.gcn_sizes,
                    activation="relu",
                    use_bias=True,
                    dropout_rate=None,
                    activate_final=False,
                    graph_pooling=True,
                    pool_type="sum")
            elif FLAGS.student_name == "gnn":
                self.student = GNN(
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
                    node_or_graph="graph")
            if self.model_name == "kd":
                teacher_trainable = False
            elif self.model_name == "dml":
                teacher_trainable = True
            if FLAGS.teacher_name == "graphsim":
                self.teacher = GraphSimNet(FLAGS, trainable=teacher_trainable)
            elif FLAGS.teacher_name == "gmn":
                self.teacher = GMNModel(FLAGS,
                                        node_or_graph="graph",
                                        trainable=teacher_trainable)
            self.kd_loss_fn = KDLoss(FLAGS)
        else:
            raise ValueError

    def call(self, graph_data, n_graphs, training):
        if self.model_name == "graphsim":
            score, _ = self.model(graph_data, n_graphs, training=training)
            rtn = {"score": score}
        elif self.model_name == "gmn":
            graph_embeddings, _ = self.model(graph_data, n_graphs, training=training)
            score = embedding_to_similarity(graph_embeddings)
            rtn = {"score": score}
        elif self.model_name == "gcn":
            graph_embeddings, _ = self.model(graph_data, training=training)
            score = embedding_to_similarity(graph_embeddings)
            rtn = {"score": score}
        elif self.model_name == "gnn":
            graph_embeddings, _ = self.model(graph_data, training=training)
            # graph_vec_scale = tf.reduce_mean(graph_embeddings ** 2)
            score = embedding_to_similarity(graph_embeddings)
            rtn = {"score": score}
                   # "gvs": graph_vec_scale}
        elif self.model_name in ["kd", "dml"]:
            graph_embeddings_s, node_states_s = self.student(
                graph_data, training=training)
            # graph_vec_scale = tf.reduce_mean(graph_embeddings_s ** 2)
            score_s = embedding_to_similarity(graph_embeddings_s)
            score_t_tmp, node_states_t = self.teacher(
                graph_data, n_graphs, training=training)
            if self.FLAGS.teacher_name == "graphsim":
                score_t = score_t_tmp
            elif self.FLAGS.teacher_name == "gmn":
                score_t = embedding_to_similarity(score_t_tmp)
            rtn = {
                "score": score_s,
                "score_s": score_s, "score_t": score_t,
                "node_states_s": node_states_s, "node_states_t": node_states_t}
                # "gvs": graph_vec_scale}
        return rtn

    def calculate_loss(self, outputs, graph_data, n_graphs, mode=None):
        if self.model_name == "dml":
            if mode == "t->s":
                score = outputs["score_s"]
            elif mode == "s->t":
                score = outputs["score_t"]
        else:
            score = outputs["score"]
        with tf.name_scope("ranking_loss"):
            y_pred = tf.reshape(
                score, [-1, 1 + self.FLAGS.n_neg_graphs])
            sim_pos, sim_neg = y_pred[:, :1], y_pred[:, 1:]
            sim_pos = tf.tile(sim_pos, [1, self.FLAGS.n_neg_graphs])
            triplet_losses = tf.reduce_sum(
                tf.nn.relu(self.FLAGS.margin - sim_pos + sim_neg),
                axis=1)
            loss = tf.reduce_sum(triplet_losses)
            acc = tf.reduce_mean(
                tf.cast(tf.greater(sim_pos, sim_neg), tf.float32))
            scores = {"acc": acc}

        if self.model_name == "dml":
            if mode == "t->s":
                weights = self.student.trainable_weights
                weight_decay = self.FLAGS.weight_decay_student
            elif mode == "s->t":
                weights = self.teacher.trainable_weights
                weight_decay = self.FLAGS.weight_decay_teacher
        else:
            weights = self.trainable_weights
            weight_decay = self.FLAGS.weight_decay
        with tf.name_scope("weight_decay_loss"):
            wdl_list = [tf.nn.l2_loss(w) for w in weights]
            wdl = tf.add_n(wdl_list)

        # if self.FLAGS.graph_vec_reg_weight == 0.0:
        #     gvs = tf.constant(0.0, dtype=tf.float32)
        # else:
        # if self.model_name in ["gnn", "kd", "dml"]:
        #     gvs = outputs["gvs"]
        # else:
        #     gvs = tf.constant(0.0, dtype=tf.float32)

        if self.model_name in ["kd", "dml"]:
            if self.model_name == "kd":
                kd_mode = "t->s"
            elif self.model_name == "dml":
                kd_mode = mode
            if self.FLAGS.kd_score_coeff == 0.0:
                kd_score_loss = tf.constant(0.0, dtype=tf.float32)
            else:
                kd_score_loss = self.kd_loss_fn(
                    outputs, graph_data, n_graphs, "score", kd_mode)
            if self.FLAGS.kd_node_coeff == 0.0:
                kd_node_loss = tf.constant(0.0, dtype=tf.float32)
            else:
                kd_node_loss = self.kd_loss_fn(
                    outputs, graph_data, n_graphs, "node", kd_mode)
            if self.FLAGS.kd_subgraph_coeff == 0.0:
                kd_subgraph_loss = tf.constant(0.0, dtype=tf.float32)
            else:
                kd_subgraph_loss = self.kd_loss_fn(
                    outputs, graph_data, n_graphs, "subgraph", kd_mode)
            # if self.model_name == "kd":
            #     total_loss = (
            #         (1 - self.FLAGS.kd_score_coeff) * loss
            #         + self.FLAGS.kd_score_coeff * kd_score_loss
            #         + self.FLAGS.kd_node_coeff * kd_node_loss
            #         + self.FLAGS.kd_subgraph_coeff * kd_subgraph_loss
            #         + weight_decay * wdl)
            #         # + self.FLAGS.graph_vec_reg_weight * 0.5 * gvs)
            # elif self.model_name == "dml":
            total_loss = (
                (1 - self.FLAGS.kd_score_coeff) * loss
                + self.FLAGS.kd_score_coeff * kd_score_loss
                + self.FLAGS.kd_node_coeff * kd_node_loss
                + self.FLAGS.kd_subgraph_coeff * kd_subgraph_loss
                + weight_decay * wdl)
                # + self.FLAGS.graph_vec_reg_weight * 0.5 * gvs)
        else:
            total_loss = (
                loss
                + weight_decay * wdl)
                # + self.FLAGS.graph_vec_reg_weight * 0.5 * gvs)

        info = {
            "losses": {
                "total_loss": total_loss,
                "loss": loss,
                "weight_decay_loss": wdl,
                "weight_decay": weight_decay
                # "gvs": gvs
            },
            "scores": scores
        }
        if self.model_name in ["kd", "dml"]:
            info["losses"]["kd_score"] = kd_score_loss
            info["losses"]["kd_node"] = kd_node_loss
            info["losses"]["kd_subgraph"] = kd_subgraph_loss
        return total_loss, info


def reshape_and_split_tensors(tensor, n_splits):
    tensor.shape.assert_has_rank(2)
    last_dim = tf.shape(tensor)[1]
    tensor = tf.reshape(tensor, [-1, last_dim * n_splits])
    return tf.split(tensor, num_or_size_splits=n_splits, axis=1)


def squared_euclidean_distance(x, y):
    assert x.shape.rank == y.shape.rank
    return tf.reduce_sum(
        tf.math.squared_difference(x, y),
        axis=-1)


def embedding_to_similarity(embeddings):
    with tf.name_scope("reshape_and_split"):
        embed_x, embed_y = reshape_and_split_tensors(embeddings, 2)
    # outputs = tf.reduce_sum(
    #     embed_x * embed_y, axis=1, keepdims=True)
    sim = - squared_euclidean_distance(embed_x, embed_y)
    outputs = tf.reshape(sim, [-1, 1])
    return outputs
