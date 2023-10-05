import collections
import numpy as np
import scipy as sp
import ray
import math
import networkx as nx
from ray.util.queue import _QueueActor
from networkx.algorithms import community

from utils.gcn_utils import preprocess_adj
from utils.common_utils import chunked


GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx",
     "node_features", "edge_features",
     "graph_idx", "n_graphs",
     "subgraph_idx", "n_subgraphs", "subgraph_to_graph_idx",
     "hatA_from_idx", "hatA_to_idx", "hatA_weights"])


SubGraphData = collections.namedtuple(
    "SubGraphData",
    ["subgraph_idx", "n_subgraphs", "subgraph_to_graph_idx"])


def batch_graphs(list_of_graph_tuple, training=True, simple=False):
    if isinstance(list_of_graph_tuple[0], nx.Graph):
        graphs = list_of_graph_tuple
    else:
        graphs = [g for tt in list_of_graph_tuple for g in tt]

    from_idx = []
    to_idx = []
    node_features = []
    edge_features = []
    graph_idx = []

    if not simple:
        hatA_from_idx = []
        hatA_to_idx = []
        hatA_weights = []
    else:
        hatA_from_idx = None
        hatA_to_idx = None
        hatA_weights = None

    n_total_nodes = 0
    n_total_edges = 0
    for i, g in enumerate(graphs):
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        edges = np.asarray(list(g.edges()))
        from_idx.append(edges[:, 0].astype(np.int32) + n_total_nodes)
        to_idx.append(edges[:, 1].astype(np.int32) + n_total_nodes)

        if not simple:
            hatA_edges = g.graph["hatA_edges"]
            hatA_from_idx.append(hatA_edges[:, 0].astype(np.int32) + n_total_nodes)
            hatA_to_idx.append(hatA_edges[:, 1].astype(np.int32) + n_total_nodes)
            hatA_weights.append(hatA_edges[:, 2].astype(np.float32))

        if g.graph.get("node_feat_type") == "onehot":
            feat_dim = g.graph["node_feat_dim"]
            feat_idx = [g.nodes[n]["feat_idx"]
                        for n in range(g.number_of_nodes())]
            feat = np.eye(feat_dim)[feat_idx].astype(np.float32)
            if not training:
                feat = sp.sparse.coo_matrix(feat)
            node_features.append(feat)
        elif g.graph.get("node_feat_type").startswith("constant"):
            feat = g.graph["node_feat"]
            node_features.append(feat)
        elif g.graph.get("node_feat_type") == "sparse":
            feat_dim = g.graph["h_dim"]
            feat = np.zeros((g.number_of_nodes(), feat_dim), dtype=np.float32)
            for n in range(g.number_of_nodes()):
                feat[n][g.nodes[n]["h"]] = 1.0
            if not training:
                feat = sp.sparse.coo_matrix(feat)
            node_features.append(feat)
        elif g.graph.get("node_feat_type") == "ready":
            feat = g.graph["node_feat"]
            node_features.append(feat)
        else:
            raise ValueError

        graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

        n_total_nodes += n_nodes
        n_total_edges += n_edges

    from_idx = np.concatenate(from_idx, axis=0)
    to_idx = np.concatenate(to_idx, axis=0)
    if sp.sparse.issparse(node_features[0]):
        node_features = sp.sparse.vstack(node_features)
    else:
        node_features = np.concatenate(node_features, axis=0)
    edge_features = np.ones((n_total_edges, 1), dtype=np.float32)
    graph_idx = np.concatenate(graph_idx, axis=0)
    n_graphs = len(graphs)

    if not simple:
        hatA_from_idx = np.concatenate(hatA_from_idx, axis=0)
        hatA_to_idx = np.concatenate(hatA_to_idx, axis=0)
        hatA_weights = np.concatenate(hatA_weights, axis=0)

        subgraph_data = batch_subgraphs(list_of_graph_tuple)
    else:
        subgraph_data = SubGraphData(
            subgraph_idx=None,
            n_subgraphs=None,
            subgraph_to_graph_idx=None)
    return GraphData(
        from_idx=from_idx, to_idx=to_idx,
        node_features=node_features, edge_features=edge_features,
        graph_idx=graph_idx, n_graphs=n_graphs,
        subgraph_idx=subgraph_data.subgraph_idx,
        n_subgraphs=subgraph_data.n_subgraphs,
        subgraph_to_graph_idx=subgraph_data.subgraph_to_graph_idx,
        hatA_from_idx=hatA_from_idx, hatA_to_idx=hatA_to_idx,
        hatA_weights=hatA_weights)


def batch_subgraphs(list_of_graph_tuple):
    if isinstance(list_of_graph_tuple[0], nx.Graph):
        graphs = list_of_graph_tuple
    else:
        graphs = [g for tt in list_of_graph_tuple for g in tt]

    subgraph_idx = []
    subgraph_to_graph_idx = []
    n_subgraphs = 0
    # k = len(graphs[0].graph["partition"])
    partiton_offset = 0
    for i, g in enumerate(graphs):
        partition = g.graph["partition"]
        # assert len(partition) == k
        n_nodes = g.number_of_nodes()
        assert int(np.sqrt(n_nodes)) == len(partition) or len(partition) in [2, 3]
        # this_idx = np.ones(n_nodes, dtype=np.int32) * i * k
        this_idx = np.ones(n_nodes, dtype=np.int32) * partiton_offset
        for j, p in enumerate(partition):
            this_idx[p] += j
            n_subgraphs += 1
        subgraph_idx.append(this_idx)
        partiton_offset += len(partition)
        subgraph_to_graph_idx.append(
            np.ones(len(partition), dtype=np.int32) * i)
    assert n_subgraphs == partiton_offset
    subgraph_idx = np.concatenate(subgraph_idx, axis=0)
    subgraph_to_graph_idx = np.concatenate(subgraph_to_graph_idx, axis=0)
    return SubGraphData(
        subgraph_idx=subgraph_idx, n_subgraphs=n_subgraphs,
        subgraph_to_graph_idx=subgraph_to_graph_idx)


batch_graphs_ray = ray.remote(num_cpus=0.0)(batch_graphs)


@ray.remote(num_cpus=0.0)
class QueueActor(_QueueActor):
    DONE = "done"


@ray.remote(num_cpus=0.0)
def parallel_batch(queue, pairs, batch_size, training, simple):
    for chunk in chunked(pairs, batch_size):
        ref = batch_graphs_ray.remote(chunk, training=training, simple=simple)
        ray.get(queue.put.remote([ref]))
    queue.put.remote(QueueActor.DONE)


def compute_hatA_edges(g):
    adj = nx.to_scipy_sparse_matrix(
        g, nodelist=range(g.number_of_nodes()))
    support = preprocess_adj(adj)
    edge_weights = np.concatenate(
        (support[0], support[1].reshape((-1, 1))),
        axis=1).astype(np.float32)
    return edge_weights


def graph_partition(g, k):
    if k == -1:
        k = int(math.sqrt(g.number_of_nodes()))
    rng = np.random.RandomState(666)
    max_modularity = -0.5
    partition = None
    for _ in range(20):
        this_partition = list(
            community.asyn_fluidc(g, k, seed=rng))
        if not community.is_partition(g, this_partition):
            continue
        this_modularity = community.modularity(g, this_partition)
        if this_modularity >= max_modularity:
            partition = this_partition
            max_modularity = this_modularity
    if partition is None:
        assert False
    partition = list(map(list, partition))
    return partition


graph_partition_ray = ray.remote(num_cpus=0.0)(graph_partition)


@ray.remote(num_cpus=0.0)
def parallel_partition(queue, graphs, k):
    for g in graphs:
        ref = graph_partition_ray.remote(g, k)
        ray.get(queue.put.remote([ref]))
    queue.put.remote(QueueActor.DONE)
