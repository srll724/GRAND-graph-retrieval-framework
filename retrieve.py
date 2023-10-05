import os
import sys
import json
import functools
import pickle
import time
import os.path as osp
import pandas as pd
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import argparse

from absl import flags
from ray.tune.result import EXPR_PARAM_FILE, EXPR_RESULT_FILE
from ray.tune.utils.util import SafeFallbackEncoder
from fwd9m.tensorflow import enable_determinism
enable_determinism()  # noqa: E402
from ray.tune.utils import date_str

from data_loader.dataset import Dataset
from model import Model, embedding_to_similarity
from utils.common_utils import DotDict, chunked
from utils.graph_utils import batch_graphs
from utils.tf_utils import pairwise_euclidean_similarity
from ranking_metrics import compute_mean as tfr_metrics
from ranking_metrics import RankingMetricKey as RMK

FLAGS = flags.FLAGS
# ffmpeg
# gcn
# flags.DEFINE_string("path", "/home/shirui/ray_results/ffmpeg/GraphSearch_ffmpeg_GRID_gcn___2022-07-15_18-51-39/Trainer_1ad2b_00000_0_ablation_dele_combine=False,batch_size=64,graphsim_encoder=gnn,kd_node_coeff=0.0,kd_score_T=1.0,kd_score_coeff=0.0,kd_subgraph_coeff=0.0,model_name=gcn,neighbor_enhanced=False,seed=6821,student_name_2022-07-15_18-51-39", "")
# gnn
# flags.DEFINE_string("path", "/home/shirui/ray_results/ffmpeg/GraphSearch_ffmpeg_GRID_gnn___2022-06-30_15-27-31/Trainer_1a368_00000_0_batch_size=64,graphsim_encoder=gnn,kd_score_T=1.0,kd_subgraph_coeff=0.0,model_name=gnn,neighbor_enhanced=False,seed=6821,student_name=gnn,teacher_name=graphsim,topk_node=False,weight_decay=0.0,weigh_2022-06-30_15-27-31", "")
# graphsim
# flags.DEFINE_string("path", "/home/shirui/ray_results/ffmpeg/GraphSearch_ffmpeg_GRID_graphsim___2022-06-29_16-42-10/Trainer_5d977_00000_0_batch_size=64,graphsim_encoder=gnn,kd_score_T=1.0,kd_subgraph_coeff=0.0,model_name=graphsim,neighbor_enhanced=False,seed=6821,student_name=gnn,teacher_name=graphsim,topk_node=False,weight_decay=0.0,_2022-06-29_16-42-10", "")
# flags.DEFINE_string("path", "/home/shirui/ray_results/ffmpeg/GraphSearch_ffmpeg_GRID_gnn___2022-06-30_15-27-31/Trainer_1a368_00000_0_batch_size=64,graphsim_encoder=gnn,kd_score_T=1.0,kd_subgraph_coeff=0.0,model_name=gnn,neighbor_enhanced=False,seed=6821,student_name=gnn,teacher_name=graphsim,topk_node=False,weight_decay=0.0,weigh_2022-06-30_15-27-31", "")

# openssl
# graphsim
# flags.DEFINE_string("path", "/home/shirui/ray_results/code/GraphSearch_code_GRID_graphsim___2022-07-18_21-01-56/Trainer_cd2bb_00000_0_ablation_dele_combine=False,batch_size=128,graphsim_encoder=gnn,kd_node_coeff=0.0,kd_score_T=1.0,kd_score_coeff=0.0,kd_subgraph_coeff=0.0,model_name=graphsim,neighbor_enhanced=False,seed=6821,studen_2022-07-18_21-01-56", "")

# cci
# graphsim
# flags.DEFINE_string("path", "/home/rshi/ray_results/cci/GraphSearch_cci_GRID_graphsim___2022-06-10_22-01-44/Trainer_dc560_00000_0_batch_size=32,kd_node_coeff=0.0,kd_score_T=1.0,kd_subgraph_coeff=0.0,model_name=graphsim,neighbor_enhanced=False,seed=6821,topk_node=False,weight_decay=0.0,weight_decay_student=0.0,weight_decay_teac_2022-06-10_22-01-44", "")

# aids new file
parser = argparse.ArgumentParser(description='Graph retrieval')
parser.add_argument('--path', type=str, default="./ray_results/aids700/GraphSearch_aids700_GRID_dml_graphsim_gnn_2022-11-02_10-35-58/Trainer_156ef_00000_0_ablation_dele_combine=False,batch_size=32,graphsim_encoder=gnn,kd_node_coeff=0.7,kd_score_T=1.0,kd_score_coeff=0.7,kd_subgraph_coeff=0.01,lr=0.001,model_name=dml,neighbor_enhanced=False,seed=6821,st_2022-11-02_10-35-58/", help='model training log directory')
parser.add_argument('--gpu', type=str, default="2", help='GPU used, such as 0,2')
parser.add_argument('--topk', type=int, default=[5, 10, 20, 50, 100, 200, 300], nargs="+",
                    help='Top-k metrics, default [5, 10, 20, 50, 100, 200, 300], input example: 5 10 20 50')
parser.add_argument('--sample', type=str, default='False', help="Retrieval sample with graph visualization")
args = parser.parse_args()

if args.sample == 'True':
    args.logdir = "./ray_results/aids700/GraphSearch_aids700_GRID_dml_graphsim_gnn_2022-11-02_10-35-58/Trainer_156ef_00000_0_ablation_dele_combine=False,batch_size=32,graphsim_encoder=gnn,kd_node_coeff=0.7,kd_score_T=1.0,kd_score_coeff=0.7,kd_subgraph_coeff=0.01,lr=0.001,model_name=dml,neighbor_enhanced=False,seed=6821,st_2022-11-02_10-35-58/"
    args.topk = [5, 10, 20, 50]

# flags.DEFINE_string("path", args.path, "")
patience = 2000
# flags.DEFINE_integer("patience", 2000, "")
# FLAGS(sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    gpu_mem_limit = 15000
    tf.config.set_visible_devices(gpu, "GPU")
    # tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu,
        [tf.config.experimental
         .VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])

with open(osp.join(args.path, EXPR_PARAM_FILE), "r") as f:
    config = json.load(f)
    config = DotDict(config)
    config.student_name = "gnn"
with open(osp.join(args.path, EXPR_RESULT_FILE), "r") as f:
    json_list = [json.loads(line) for line in f if line]
df = pd.json_normalize(json_list, sep="/")
if config.model_name in ["graphsim", "gmn"]:
    df = df[df.training_iteration <= 120000]
    df = df.loc[df["searcher_metric"].notna()]
    df["metric"] = df.filter(
        regex=f"val/scores/top_5/(ndcg|recall)").mean(axis=1)
else:
    df = df[df.training_iteration <= 120000]
    df = df.loc[df["searcher_metric"].notna()]
    df["metric"] = df.filter(
        regex=f"val/scores/top_20/(ndcg|recall)").mean(axis=1)
best = 0.0
wait = patience
for i, value in enumerate(df.metric):
    if value < best:
        wait -= 1
    else:
        best = value
        wait = patience
    if wait == 0:
        break
df = df.iloc[:i + 1]

idx = df["metric"].idxmax()
best_iter = df.loc[idx]["training_iteration"]
print(idx, best_iter)

if config.model_name in ["graphsim", "gmn"]:
    test_mode = "retrieve"
else:
    test_mode = "test_ebp"
ds = Dataset.instantiate(
    name=config.data_name,
    node_ordering=config.node_ordering,
    n_subgraphs=config.n_subgraphs,
    batch_size=config.batch_size,
    n_neg_graphs=config.n_neg_graphs,
    # batch_size_eval=1,
    batch_size_eval=config.batch_size_eval,
    seed=config.seed,
    mode=test_mode)

model = Model(config)
dummy_data = ds.get_data(mode="train")
_ = model(dummy_data["graphs"], ds.n_graphs_per_batch_train, training=True)

if config.model_name == "dml":
    ckpt_kwargs = {"dml": model}
elif config.model_name in ["graphsim", "gcn", "gnn", "gmn"]:
    ckpt_kwargs = {config.model_name: model.model}
else:
    raise NotImplementedError
ckpt = tf.train.Checkpoint(**ckpt_kwargs)
ckpt.restore(
    osp.join(args.path, "checkpoints", f"ckpt-{best_iter}")).assert_consumed()

# assert False
if config.model_name in ["graphsim", "gmn"]:
    if config.model_name == "dml":
        fn = model.teacher
    else:
        fn = model
    def forward_fn_factory(n_graphs):
        func = functools.partial(
            fn, n_graphs=n_graphs, training=False)
        return tf.function(func, input_signature=[ds.graph_signature])

    forward_fn_cache = functools.lru_cache(maxsize=None)(forward_fn_factory)

    # _ = forward_fn_cache(
    #     n_graphs=int(ds.test_mbp_data[0].n_graphs.numpy()))(
    #     process_fn(ds.test_mbp_data[0]))
    # _ = forward_fn_cache(
    #     n_graphs=int(ds.test_mbp_data[-1].n_graphs.numpy()))(
    #     process_fn(ds.test_mbp_data[-1]))

    y_pred = []
    start = time.time()
    for d in ds.get_data("test"):
        d = d["graphs"]
        n_graphs = int(d.n_graphs.numpy())
        outputs = forward_fn_cache(n_graphs=n_graphs)(d)
        if config.model_name == "dml":
            if config.teacher_name == "graphsim":
                score = outputs[0]
            elif config.teacher_name == "gmn":
                score = embedding_to_similarity(outputs[0])
        else:
            score = outputs["score"]
        y_pred.append(tf.reshape(score, [-1]))

        if n_graphs != 2048:
            print("query consume time :", time.time()-start)

        if len(y_pred) % 1000 == 0:
            print(f"len(y_pred) = {len(y_pred)}, elapsed {time.time() - start}s")
        if config.data_name == "cci":
            if sum([x.shape[0] for x in y_pred]) > 5005 * len(ds.graphs):
                break
    # 新增metric部分
    eval_metric_keys = [RMK.NDCG, RMK.RECALL, RMK.PRECISION, RMK.MRR, RMK.MAP]
    topks = args.topk
    labels = getattr(ds, "test_mbp_labels")
    predictions = tf.reshape(tf.concat(y_pred, axis=0), labels.shape)
    # if self.FLAGS.data_name == "aids700":
    #     predictions = self.ds.handle_ebp_predictions(predictions, 0, mode)
    info = {}
    for k in topks:
        info[f"top_{k}"] = {
            key: tfr_metrics(key, labels, predictions, topn=k)
            for key in eval_metric_keys}
        print("info this model:", info)
    # ######################################################


    # dirname = f"./test_mbp_results/{config.data_name}"
    # os.system(f"mkdir -p {dirname}")
    # teacher_name = config.get("teacher_name", None)
    # student_name = config.get("student_name", None)
    # with open(f"{dirname}/{config.model_name}_{teacher_name}_{student_name}_{config.seed}.pkl", "wb") as f:
    #     pickle.dump(y_pred, f)

    # y_pred = tf.reshape(
    #     tf.concat(y_pred, axis=0), [-1, ds.test_mbp_labels.shape[1]])

    # predictions = y_pred.numpy()
    # offset = len(ds.train_graphs) + len(ds.val_graphs)
    # for i in range(predictions.shape[0]):
    #     predictions[i, offset + i] = - 9999.0
    # predictions = tf.convert_to_tensor(predictions)
    # labels = ds.test_mbp_labels

else:
    eval_metric_keys = [RMK.NDCG, RMK.RECALL, RMK.MRR, RMK.MAP]
    if config.model_name in ["dml", "kd"]:
        fn = model.student
    elif config.model_name in ["gnn", "gcn"]:
        fn = model.model
    mode = "test"
    query_data = getattr(ds, f"{mode}_ebp_query_data")
    candidate_data = getattr(ds, f"{mode}_ebp_candidate_data")
    labels = getattr(ds, f"{mode}_ebp_labels")

    # 把合起来的数据分成一个一个图的形式，转换为nx格式，画图
    if args.sample == 'True':
        index_idx = 0
        num_graph_node = []
        num_graph_edge = []
        for gid in range(query_data[0].n_graphs):
            graph_node_len = len(query_data[0].graph_idx[query_data[0].graph_idx == gid])
            graph_edge_len = len(query_data[0].from_idx[(query_data[0].from_idx - index_idx) < graph_node_len]) - sum(num_graph_edge)
            num_graph_node.append(graph_node_len)
            num_graph_edge.append(graph_edge_len)
            index_idx = index_idx + graph_node_len
        node_id = tf.convert_to_tensor(range(sum(num_graph_node)))
        graph_node_id = tf.split(node_id, num_graph_node, axis=0)
        graph_node_feature = tf.split(query_data[0].node_features, num_graph_node, axis=0)
        graph_edge_from_id = tf.split(query_data[0].from_idx, num_graph_edge, axis=0)
        graph_edge_to_id = tf.split(query_data[0].to_idx, num_graph_edge, axis=0)

        graph_edge_id = []
        pair = []
        for gid in range(query_data[0].n_graphs):
            for from_id, to_id in zip(graph_edge_from_id[gid], graph_edge_to_id[gid]):
                pair.append((int(from_id), int(to_id)))
            graph_edge_id.append(pair)
            pair = []

        query_graph_G = []
        for gid in range(query_data[0].n_graphs):
            G = nx.Graph()
            G.add_nodes_from(graph_node_id[gid].numpy().tolist(), weight=graph_node_feature[gid].numpy().tolist())
            G.add_edges_from(graph_edge_id[gid])
            query_graph_G.append(G)

        index_idx = 0
        num_graph_node = []
        num_graph_edge = []
        for gid in range(candidate_data[0].n_graphs):
            graph_node_len = len(candidate_data[0].graph_idx[candidate_data[0].graph_idx == gid])
            graph_edge_len = len(candidate_data[0].from_idx[(candidate_data[0].from_idx - index_idx) < graph_node_len]) - sum(
                num_graph_edge)
            num_graph_node.append(graph_node_len)
            num_graph_edge.append(graph_edge_len)
            index_idx = index_idx + graph_node_len
        node_id = tf.convert_to_tensor(range(sum(num_graph_node)))
        graph_node_id = tf.split(node_id, num_graph_node, axis=0)
        graph_node_feature = tf.split(candidate_data[0].node_features, num_graph_node, axis=0)
        graph_edge_from_id = tf.split(candidate_data[0].from_idx, num_graph_edge, axis=0)
        graph_edge_to_id = tf.split(candidate_data[0].to_idx, num_graph_edge, axis=0)

        graph_edge_id = []
        pair = []
        for gid in range(candidate_data[0].n_graphs):
            for from_id, to_id in zip(graph_edge_from_id[gid], graph_edge_to_id[gid]):
                pair.append((int(from_id), int(to_id)))
            graph_edge_id.append(pair)
            pair = []

        candidate_graph_G = []
        for gid in range(candidate_data[0].n_graphs):
            G = nx.Graph()
            G.add_nodes_from(graph_node_id[gid].numpy().tolist(), weight=graph_node_feature[gid].numpy().tolist())
            G.add_edges_from(graph_edge_id[gid])
            candidate_graph_G.append(G)

    candidate_embeddings = []
    for data in candidate_data:
        embeddings, _ = fn(data, training=False)
        candidate_embeddings.append(embeddings)
    candidate_embeddings = tf.concat(candidate_embeddings, axis=0)

    info_list = []
    weights = []
    for i, data in enumerate(query_data):
        start = time.time()
        embeddings, _ = fn(data, training=False)
        predictions = pairwise_euclidean_similarity(
            embeddings, candidate_embeddings)
        predictions = ds.handle_ebp_predictions(predictions, i, mode)
        ### topk G
        if args.sample == 'True':
            ranked_predictions = tf.math.top_k(predictions, k=5) # args.topk
            ### 可视化部分
            n = 0
            for one_query_id in range(data.n_graphs):
                true_samples = 0
                topk_graph_id = ranked_predictions.indices[one_query_id]
                plt.figure(figsize=(36, 9))
                plt.suptitle('Chemical compounds similar search in AIDS', fontsize='xx-large', fontweight='heavy')
                plt.subplot(1, 6, 1)
                plt.title('Query chemical compound')
                nx.draw(query_graph_G[one_query_id], pos=nx.spectral_layout(query_graph_G[one_query_id]))
                for k in range(5): # args.topk
                    topk_graph_G = candidate_graph_G[topk_graph_id[k]]
                    plt.subplot(1, 6, k+2)
                    if labels[0][one_query_id, topk_graph_id[k]] == 1:
                        plt.title("Target chemical compound")
                        nx.draw(topk_graph_G, node_color='r', pos=nx.spectral_layout(topk_graph_G))
                        true_samples = 1
                    else:
                        plt.title("rank "+f"{k+1}")
                        nx.draw(topk_graph_G)
                if true_samples == 1:
                    plt.savefig(f'./log/retrieve/graph_vis/query_graph_{n}.png')
                    # plt.show()
                    n += 1
                elif true_samples == 0:
                    plt.clf()
                # print('num_querys: ', n)
                # plt.savefig(re)
        print("one query time(ms): ", (time.time() - start) * 1000)
        this_info = {}
        for k in args.topk:
            this_info[f"top_{k}"] = {
                key: tfr_metrics(key, labels[i], predictions, topn=k)
                for key in eval_metric_keys}
        info_list.append(this_info)
        weights.append(labels[i].shape[0])
    info_tensor = tf.convert_to_tensor(
        [tf.nest.flatten(x) for x in info_list])
    weights = tf.cast(
        tf.convert_to_tensor(weights)[:, tf.newaxis], tf.float32)
    info_tensor = tf.reduce_sum(weights * info_tensor, axis=0)
    info = tf.nest.pack_sequence_as(
        this_info,
        tf.unstack(info_tensor / tf.reduce_sum(weights)))
    log = open(f"./log/retrieve/graph_info/GraphSearch_{args.topk}_{date_str()}.txt", mode="w", encoding="utf-8")
    print("info this model:", info, file=log)
    log.close()

    fig_path = os.path.abspath(f"./log/retrieve/graph_vis/")
    path = os.path.abspath(f"./log/retrieve/graph_info/GraphSearch_{args.topk}_{date_str()}.txt")
    print("\n Retrieval log is written to: ", path, "\n")
    if args.sample == 'True':
        print("Retrieval visualization results are written to: ", fig_path, "\n")
    # # candidate_graphs = ds.train_graphs + ds.val_graphs + ds.test_graphs
    # # query_graphs = ds.test_graphs

    # # batch_candidate_graphs = [
    # #     batch_graphs(chunk, training=False)
    # #     for chunk in chunked(candidate_graphs, ds.batch_size_eval)]
    # # batch_query_graphs = [
    # #     batch_graphs(chunk, training=False)
    # #     for chunk in chunked(query_graphs, ds.batch_size_eval)]

    # batch_candidate_graphs = ds.test_candidate_data
    # batch_query_graphs = ds.test_query_data
    # labels = ds.test_labels

    # candidate_embeddings = []
    # for data in batch_candidate_graphs:
    #     embeddings, _ = model.student(data, training=False)
    #     candidate_embeddings.append(embeddings)
    # candidate_embeddings = tf.concat(candidate_embeddings, axis=0)
    # query_embeddings = []
    # for data in batch_query_graphs:
    #     embeddings, _ = model.student(data, training=False)
    #     query_embeddings.append(embeddings)
    # query_embeddings = tf.concat(query_embeddings, axis=0)

    # preds = pairwise_euclidean_similarity(
    #     query_embeddings, candidate_embeddings)
    # predictions = ds.mask_test_predictions(preds)
    # # offset = len(ds.train_graphs) + len(ds.val_graphs)
    # # for i in range(len(query_graphs)):
    # #     preds[i, i + offset] = - 9999.
    # # preds = tf.convert_to_tensor(preds)
    # # label_indices = []
    # # label_values = []
    # # for gid1, gid2 in ds.test_pos_pairs:
    # #     index_1 = ds.test_gid_to_index[gid1]
    # #     a = ds.train_gid_to_index.get(gid2)
    # #     b = ds.val_gid_to_index.get(gid2)
    # #     c = ds.test_gid_to_index.get(gid2)
    # #     if a is not None:
    # #         assert False
    # #         index_2 = a
    # #     if b is not None:
    # #         assert False
    # #         index_2 = b + len(ds.train_graphs)
    # #     if c is not None:
    # #         index_2 = c + len(ds.train_graphs) + len(ds.val_graphs)
    # #     label_indices.append((index_1, index_2))
    # #     label_values.append(1.0)
    # # labels = tf.SparseTensor(label_indices, label_values, preds.shape)
    # # labels = tf.sparse.to_dense(tf.sparse.reorder(labels))

# keys = [RMK.MAP, RMK.MRR, RMK.NDCG, RMK.RECALL]
# results = {"seed": config.seed}
# for k in [50, 100, 200, 300, 500]:
#     print(k)
#     for key in keys:
#         score = tfr_metrics(key, labels, predictions, topn=k).numpy()
#         results[f"{k}_{key}"] = score
# name = os.path.basename(FLAGS.path.strip("/"))
# print(name)
# print(results)
# with open(f"results_{name}.json", "a") as f:
#     json.dump(results, f, cls=SafeFallbackEncoder)
#     f.write("\n")
