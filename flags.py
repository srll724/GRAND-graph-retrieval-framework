import sys

from utils.flag_utils import flags, flag_dict


flags.DEFINE_string("DUMMY_FLAG", "", "")

flags.DEFINE_integer("seed", 42, "")
flags.DEFINE_integer("gpu", 1, "")
flags.DEFINE_string("tune_mode", "GRID", "")
# flags.DEFINE_string("tune_mode", "ASHA", "****************")
flags.DEFINE_integer("gcn_mode", 2, "")

flags.DEFINE_string("data_name", "code", "")
flags.DEFINE_string("node_ordering", "bfs", "")
# flags.DEFINE_float("valid_percentage", 0.25, "")
flags.DEFINE_integer("n_subgraphs", -1, "****************")

# flags.DEFINE_string("model_name", "graphsim", "")
# flags.DEFINE_string("model_name", "gcn", "")
# flags.DEFINE_string("model_name", "gnn", "")
# flags.DEFINE_string("model_name", "kd", "")
flags.DEFINE_string("model_name", "dml", "")
flags.DEFINE_string("student_name", "gnn", "")
flags.DEFINE_string("teacher_name", "graphsim", "")
flags.DEFINE_boolean("write_graph", False, "")
flags.DEFINE_boolean("checkpoint", True, "")

# ============ Graph Similarity Network ============
flags.DEFINE_boolean("use_all_node_states", False, "")
flags.DEFINE_list_integer("gcn_sizes",
                          [128, 128, 128], "")
flags.DEFINE_integer("fixed_size", 54, "")
# flags.DEFINE_integer("fixed_size", 108, "")
flags.DEFINE_list_integer("cnn_filter_sizes",
                          [16, 32, 64, 128, 128], "")
                          # [16, 32, 64, 128, 256, 512], "")
flags.DEFINE_list_integer("cnn_kernel_sizes",
                          [6, 6, 5, 5, 5], "")
                          # [5, 5, 5, 5, 5, 5], "")
flags.DEFINE_list_integer("cnn_pool_sizes",
                          [2, 2, 2, 3, 3], "")
                          # [2, 2, 2, 2, 3, 3], "")
flags.DEFINE_list_integer("mlp_sizes",
                          [256, 128, 64, 16, 1], "")
                          # [256, 128, 64, 32, 16, 8, 1], "")

flags.DEFINE_boolean("clip_grad", True, "")
flags.DEFINE_float("grad_clip_value", 10.0, "")
flags.DEFINE_float("weight_decay", 0.0, "****************")
flags.DEFINE_float("weight_decay_teacher", 1e-4, "")
flags.DEFINE_float("weight_decay_student", 1e-5, "")
flags.DEFINE_float("margin", 1.0, "")
# flags.DEFINE_float("kd_rank_coeff", 1.0, "")
flags.DEFINE_float("kd_score_T", 1.0, "")
flags.DEFINE_float("kd_score_coeff", 0.0, "")
flags.DEFINE_integer("kd_node_mode", 1, "****************")
flags.DEFINE_float("kd_node_T", 1.0, "")
flags.DEFINE_float("kd_node_coeff", 0.0, "")
flags.DEFINE_boolean("topk_node", False, "")
flags.DEFINE_integer("kd_subgraph_mode", 1, "****************")
flags.DEFINE_float("kd_subgraph_T", 1.0, "")
flags.DEFINE_float("kd_subgraph_coeff", 0.0, "")
flags.DEFINE_string("segment", "mean", "")
flags.DEFINE_boolean("neighbor_enhanced", False, "****************")

flags.DEFINE_integer("batch_size", 64, "")
flags.DEFINE_integer("n_neg_graphs", 1, "")
flags.DEFINE_integer("batch_size_eval", 1024, "")
# flags.DEFINE_integer("batch_size_retrieval", 1024, "")
# flags.DEFINE_integer("validation_size", 5000, "")
# flags.DEFINE_integer("test_size", 5000, "")
flags.DEFINE_float("lr", 1e-3, "")
flags.DEFINE_integer("n_training_steps", 120000, "****************")
flags.DEFINE_integer("eval_period", 3000, "****************")
# flags.DEFINE_integer("patience", 10, "")
# ==================================================

# ============ Graph Matching Network ============
flags.DEFINE_integer("node_state_dim", 32, "")
flags.DEFINE_integer("graph_state_dim", 128, "")

# graph encoder
flags.DEFINE_list_integer("ge_node_hidden_sizes", [32], "")
flags.DEFINE_list_integer("ge_edge_hidden_sizes", None, "")

# graph aggregator
flags.DEFINE_list_integer("ga_node_hidden_sizes", [128], "")
flags.DEFINE_list_integer("graph_hidden_sizes", [128], "")
flags.DEFINE_boolean("gated", True, "")
flags.DEFINE_string("aggregation_type", "sum", "")

# graph prop layer
flags.DEFINE_list_integer("gp_edge_hidden_sizes", [64, 64], "")
flags.DEFINE_list_integer("gp_node_hidden_sizes", [64], "")
flags.DEFINE_float("edge_net_init_scale", 0.1, "")
flags.DEFINE_string("node_update_type", "gru", "")
flags.DEFINE_boolean("use_reverse_direction", True, "")
flags.DEFINE_boolean("reverse_param_different", False, "")
flags.DEFINE_boolean("use_layer_norm", False, "")

# graph embedding net / graph matching net
flags.DEFINE_string("model_type", "embedding", "")
flags.DEFINE_integer("n_prop_layers", 5, "")
flags.DEFINE_boolean("share_prop_params", True, "")
flags.DEFINE_string("matching_similarity_type", "dotproduct", "")

# training
# flags.DEFINE_string("training_mode", "triplet", "")
# flags.DEFINE_integer("batch_size", 10, "")
# flags.DEFINE_integer("validation_size", 1000, "")
# flags.DEFINE_integer("test_size", 1000, "")
# flags.DEFINE_float("lr", 1e-3, "")
# flags.DEFINE_float("margin", 1.0, "")
flags.DEFINE_float("graph_vec_reg_weight", 0.0, "")
# flags.DEFINE_boolean("clip_grad", True, "")
# flags.DEFINE_float("grad_clip_value", 10.0, "")
# flags.DEFINE_integer("n_training_steps", 500000, "")
# flags.DEFINE_integer("n_substeps", 100, "")
# flags.DEFINE_integer("eval_after", 10, "")
# flags.DEFINE_integer("patience", 10, "")
# ================================================

flags.DEFINE_boolean("autograph", True, "")

flags.DEFINE_string("graphsim_encoder", 'gnn', "")
flags.DEFINE_boolean("ablation_dele_combine", False, "")

# flags.FLAGS(sys.argv)

FLAGS = flag_dict()
