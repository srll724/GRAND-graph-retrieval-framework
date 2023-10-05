import os.path as osp
import itertools
import functools
import pickle
import networkx as nx
import numpy as np
import scipy as sp
import ray
import tensorflow as tf

from utils.common_utils import chunked, string_to_numpy
from utils.node_ordering import node_ordering as node_ordering_func
from utils.graph_utils import (GraphData, batch_graphs,
                               QueueActor, parallel_batch,
                               compute_hatA_edges, graph_partition)
from utils.dataset_utils import control_rng


class Dataset:
    def __init__(self,
                 node_ordering,
                 n_subgraphs,
                 batch_size,
                 n_neg_graphs,
                 batch_size_eval,
                 seed,
                 mode="train"):
        self.node_ordering = node_ordering
        self.n_subgraphs = n_subgraphs
        self.batch_size = batch_size
        self.n_neg_graphs = n_neg_graphs
        self.batch_size_eval = batch_size_eval
        self.seed = seed

        self.n_graphs_per_batch_train = (
            self.batch_size * 2 * (1 + self.n_neg_graphs))

        self.rng = np.random.RandomState(self.seed)

        self.init()
        self.load_data()
        self.preprocess_graphs()
        if mode == "train":
            self.prepare_batched_mbp("val")
            self.prepare_batched_ebp("val")
            self.prepare_batched_ebp("test")
            if self.data_name == "aids700":
                self.prepare_batched_mbp("test")
        elif mode == "test_mbp":
            self.prepare_batched_mbp("test")
        elif mode == "test_ebp":
            self.prepare_batched_ebp("test")
            # self.prepare_batched_test()
        elif mode == "retrieve":
            self._mode = mode
        else:
            assert False
        # if self.data_name in ["aids700"]:
        # self.prepare_batched_val_test(mode="test")

    @staticmethod
    def instantiate(name, **kwargs):
        if name == "aids700":
            from data_loader.aids_dataset import AIDSDataset
            cls = AIDSDataset
        elif name == "imdb":
            from data_loader.imdb_dataset import IMDBDataset
            cls = IMDBDataset
        elif name == "linux":
            from data_loader.linux_dataset import LinuxDataset
            cls = LinuxDataset
        elif name == "cci":
            from data_loader.cci_dataset import CCIDataset
            cls = CCIDataset
        elif name == "code":
            from data_loader.code_dataset import CodeDataset
            cls = CodeDataset
        elif name == "alchemy":
            from data_loader.alchemy_dataset import AlchemyDataset
            cls = AlchemyDataset
        elif name == "ffmpeg":
            from data_loader.ffmpeg_dataset import FfmpegDataset
            cls = FfmpegDataset
        return cls(**kwargs)

    def init(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError

    def prepare_val(self):
        raise NotImplementedError

    def prepare_test(self):
        raise NotImplementedError

    def sample_once(self, rng):
        raise NotImplementedError

    def _sample_once(self, rng):
        raise NotImplementedError

    def get_data(self, mode):
        if mode == "train":
            if not hasattr(self, "train_iterator"):
                train_tfds = self.build_tf_dataset(mode="train")
                self.train_iterator = iter(train_tfds)
            return next(self.train_iterator)
        elif mode in ["val", "test"]:
            assert mode == "test"
            if mode == "test":
                assert self._mode == "retrieve"
            # if self.data_name in ["aids700"]:
            if not hasattr(self, f"{mode}_tfds"):
                tfds = self.build_tf_dataset(mode=mode)
                setattr(self, f"{mode}_tfds", tfds)
            return getattr(self, f"{mode}_tfds")
            # elif self.data_name == "cci":
            #     return self.eval_generator_ray(mode)

    @property
    def graph_signature(self):
        if hasattr(self, "_graph_signature"):
            return self._graph_signature
        # if self.node_feat_type == "onehot":
        #     # node_features = tf.SparseTensorSpec(
        #     #     shape=[None, self.node_feat_dim], dtype=tf.float32)
        #     node_features = tf.TensorSpec(
        #         shape=[None, self.node_feat_dim], dtype=tf.float32)
        graph_signature = GraphData(
            from_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            to_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            node_features=tf.TensorSpec(
                shape=[None, self.node_feat_dim], dtype=tf.float32),
            edge_features=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
            graph_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            n_graphs=tf.TensorSpec(shape=[], dtype=tf.int32),
            subgraph_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            n_subgraphs=tf.TensorSpec(shape=[], dtype=tf.int32),
            subgraph_to_graph_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            hatA_from_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            hatA_to_idx=tf.TensorSpec(shape=[None], dtype=tf.int32),
            hatA_weights=tf.TensorSpec(shape=[None], dtype=tf.float32))
        self._graph_signature = graph_signature
        return self._graph_signature

    @control_rng
    def sample_train_pairs(self, rng):
        graph_pairs = []
        for _ in range(self.batch_size):
            query, pos, k_neg = self.sample_once(rng)
            pos = [pos]
            graph_pairs.append(
                sum([(query, g) for g in itertools.chain(pos, k_neg)], ()))
        return graph_pairs

    def train_generator(self):
        rng = np.random.RandomState(self.seed + 1)
        while True:
            graphs = self.sample_train_pairs(rng=rng)
            this_batch = {
                "graphs": batch_graphs(graphs)}
            yield this_batch

    def eval_generator(self, mode):
        assert mode == "test"
        def process_fn(data):
            data = data._replace(
                subgraph_idx=tf.constant([-1], dtype=tf.int32),
                n_subgraphs=tf.constant(-1, dtype=tf.int32),
                subgraph_to_graph_idx=tf.constant([-1], dtype=tf.int32),
                hatA_from_idx=tf.constant([-1], dtype=tf.int32),
                hatA_to_idx=tf.constant([-1], dtype=tf.int32),
                hatA_weights=tf.constant([-1.0], dtype=tf.float32))
            return data
        for qg in self.test_mbp_query_graphs:
            for cgs in chunked(self.test_mbp_candidate_graphs, self.batch_size_eval):
                pairs = [(qg, cg) for cg in cgs]
                d = batch_graphs(pairs, training=True, simple=True)
                d = process_fn(d)
                # if sp.sparse.issparse(d.node_features):
                #     d = d._replace(node_features=d.node_features.toarray())
                yield {"graphs": d}

    def prepare_batched_mbp(self, mode):
        assert mode in ["val", "test"]
        suffix = self.batched_data_file_suffix
        filepath = osp.join(self.data_dir,
                            f"batched_{mode}_mbp_data_{suffix}.pkl")
        if osp.isfile(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                labels = pickle.load(f)
        else:
            graph_pairs, labels = self.prepare_mbp(mode)
            use_ray = True
            if mode == "test":
                if self.data_name == "cci":
                    batch_training = False
                    batch_simple = True
                else:
                    batch_training = True
                    batch_simple = False
            else:
                batch_training = True
                batch_simple = False
            if use_ray:
                if not ray.is_initialized():
                    ray.init()
                queue = QueueActor.remote(maxsize=10)
                parallel_batch.remote(
                    queue, graph_pairs, self.batch_size_eval, training=batch_training, simple=batch_simple)
                data = []
                while True:
                    res = ray.get(queue.get.remote())
                    if res == QueueActor.DONE:
                        ray.kill(queue)
                        break
                    else:
                        data.append(ray.get(res[0]))
            else:
                assert False
                graph_pairs = iter(graph_pairs)
                data = [
                    batch_graphs(chunk, training=False)
                    for chunk in chunked(graph_pairs, self.batch_size_eval)]
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
                pickle.dump(labels, f)
        size = labels.shape.num_elements()
        assert len(data) == size // self.batch_size_eval + 1 # TODO: 如果这里刚好整除了，这个assert好像就不能满足(需要看一下batch的处理部分)

        def func(t):
            try:
                t = tf.convert_to_tensor(t)
            except:
                t = t
            return t
        data = tf.nest.map_structure(func, data)
        setattr(self, f"{mode}_mbp_data", data)
        setattr(self, f"{mode}_mbp_labels", labels)

    def prepare_batched_ebp(self, mode):
        assert mode in ["val", "test"]
        suffix = self.batched_data_file_suffix
        filepath = osp.join(
            self.data_dir, f"batched_{mode}_ebp_data_{suffix}.pkl")
        if osp.isfile(filepath):
            with open(filepath, "rb") as f:
                batched_query_graphs = pickle.load(f)
                batched_candidate_graphs = pickle.load(f)
                labels = pickle.load(f)
        else:
            query_graphs, candidate_graphs, labels = self.prepare_ebp(mode)
            batched_query_graphs = [
                batch_graphs(chunk, training=True)
                for chunk in chunked(query_graphs, self.batch_size_eval)]
            batched_candidate_graphs = [
                batch_graphs(chunk, training=True)
                for chunk in chunked(candidate_graphs, self.batch_size_eval)]
            with tf.device("/device:cpu:0"):
                labels = tf.split(
                    labels,
                    num_or_size_splits=[
                        d.n_graphs for d in batched_query_graphs],
                    axis=0)
                labels = tf.nest.map_structure(tf.sparse.from_dense, labels)
            with open(filepath, "wb") as f:
                pickle.dump(batched_query_graphs, f)
                pickle.dump(batched_candidate_graphs, f)
                pickle.dump(labels, f)
        setattr(self, f"{mode}_ebp_query_data",
                tf.nest.map_structure(
                    tf.convert_to_tensor, batched_query_graphs))
        setattr(self, f"{mode}_ebp_candidate_data",
                tf.nest.map_structure(
                    tf.convert_to_tensor, batched_candidate_graphs))
        if isinstance(labels, list):
            assert isinstance(labels[0], tf.sparse.SparseTensor)
            labels = tf.nest.map_structure(
                lambda x: tf.sparse.to_dense(tf.sparse.reorder(x)), labels)
        setattr(self, f"{mode}_ebp_labels", labels)

    def eval_generator_ray(self, mode):
        if not ray.is_initialized():
            ray.init()
        # if not hasattr(self, "subgraph_dict_ref"):
        #     self.subgraph_dict_ref = ray.put(self.subgraph_dict)
        # if not hasattr(self, "hatA_graphs_ref"):
        #     self.hatA_graphs_ref = ray.put(self.hatA_graphs)
        if not hasattr(self, f"{mode}_pairs_ref"):
            setattr(self, f"{mode}_pairs_ref",
                    ray.put(getattr(self, f"{mode}_pairs")))
        graph_pairs_ref = getattr(self, f"{mode}_pairs_ref")

        queue = QueueActor.remote(maxsize=10)
        parallel_batch.remote(
            queue, graph_pairs_ref, self.batch_size_eval)

        while True:
            res = ray.get(queue.get.remote())
            if res == QueueActor.DONE:
                ray.kill(queue)
                return
            else:
                graphs = ray.get(res[0])
                graphs = tf.nest.map_structure(tf.convert_to_tensor, graphs)
                yield {"graphs": graphs}

    def test_epr_time(self):
        import time
        for data in self.eval_generator_ray("val"):
            time.sleep(0.05)

    def build_tf_dataset(self, mode):
        if mode == "train":
            generator = self.train_generator
        else:
            assert mode == "test"
            self.test_mbp_query_graphs, self.test_mbp_candidate_graphs, self.test_mbp_labels = self.prepare_mbp("test")
            generator = functools.partial(self.eval_generator, mode=mode)
        output_signature = {"graphs": self.graph_signature}
        tf_dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature)
        tf_dataset = tf_dataset.prefetch(1)
        return tf_dataset

    def preprocess_graphs(self):
        self.raw_train_graphs = self.train_graphs
        self.raw_val_graphs = self.val_graphs
        self.raw_test_graphs = self.test_graphs

        # if self.node_ordering is not None:
        #     self.relabel_nodes()
        self.assign_node_features()
        self.construct_hatA_graphs()
        self.partition_graphs()

    def relabel_nodes(self):
        assert False
        def _relabeling_func(g):
            if self.node_ordering is None:
                assert False
                new_g = nx.convert_node_labels_to_integers(g)
            elif self.node_ordering == "bfs":
                order, mapping = node_ordering_func(
                    g, "bfs", self.node_feat_name, [])
                node_list = list(g.nodes)
                mapping = {node_list[j]: i for i, j in enumerate(order)}
                new_g = nx.relabel_nodes(g, mapping)
            else:
                raise ValueError
            return new_g
        self.train_graphs = list(map(_relabeling_func, self.train_graphs))
        self.val_graphs = list(map(_relabeling_func, self.val_graphs))
        self.test_graphs = list(map(_relabeling_func, self.test_graphs))

    def assign_node_features(self):
        if self.node_feat_type == "onehot":
            feat_set = set()
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                node_attr = nx.get_node_attributes(g, self.node_feat_name)
                assert len(node_attr) == g.number_of_nodes()
                feat_set = feat_set.union(node_attr.values())
            feat_mapping = {feat: idx
                            for idx, feat in enumerate(sorted(
                                feat_set,
                                key=lambda x: "None" if x is None else x))}
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                for n in g.nodes():
                    g.nodes[n]["feat_idx"] = feat_mapping[
                        g.nodes[n][self.node_feat_name]]
                g.graph["node_feat_type"] = self.node_feat_type
                g.graph["node_feat_dim"] = len(feat_set)
            self.node_feat_dim = len(feat_set)
        elif self.node_feat_type.startswith("constant"):
            input_dim = int(self.node_feat_type.split("_")[1])
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                g.graph["node_feat_type"] = self.node_feat_type
                g.graph["node_feat"] = np.full(
                    (g.number_of_nodes(), input_dim), 2.0, dtype=np.float32)
            self.node_feat_dim = input_dim
        elif self.node_feat_type == "degree":
            feat_set = set()
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                node_degrees = dict(g.degree)
                feat_set = feat_set.union(node_degrees.values())
            feat_mapping = {feat: idx
                            for idx, feat in enumerate(sorted(feat_set))}
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                for n in g.nodes():
                    g.nodes[n]["feat_idx"] = feat_mapping[g.degree[n]]
                g.graph["node_feat_type"] = "onehot"
                g.graph["node_feat_dim"] = len(feat_set)
            self.node_feat_dim = len(feat_set)
        elif self.node_feat_type == "sparse":
            self.node_feat_dim = self.train_graphs[0].graph["h_dim"]
        elif self.node_feat_type == "ready":
            for g in itertools.chain(self.train_graphs, self.val_graphs,
                                     self.test_graphs):
                if "node_feat" in g.graph:
                    assert False
                    node_feat = string_to_numpy(g.graph["node_feat"])
                    g.graph["node_feat"] = node_feat
                    g.graph["node_feat_type"] = "ready"
                else:
                    node_feat = [
                        g.nodes[n]["feat"] for n in range(g.number_of_nodes())]
                    node_feat = np.array(node_feat, dtype=np.float32)
                    g.graph["node_feat"] = node_feat
                    g.graph["node_feat_type"] = "ready"
            self.node_feat_dim = node_feat.shape[1]
        else:
            raise ValueError

    def construct_hatA_graphs(self):
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["hatA_edges"] = compute_hatA_edges(g)

    def partition_graphs(self):
        for g in itertools.chain(
                self.train_graphs, self.val_graphs, self.test_graphs):
            g.graph["partition"] = graph_partition(g, self.n_subgraphs)


if __name__ == "__main__":
    ds = Dataset(
        data_name="aids700",
        task="ranking",
        node_ordering="bfs",
        node_feat_type="onehot",
        node_feat_name="type",
        valid_percentage=0.25,
        batch_size=128,
        n_neg_graphs=5,
        batch_size_eval=256,
        batch_size_retrieval=512,
        validation_size=1000,
        test_size=1000,
        seed=123)
    # data = ds.get_pairs(128)
    # data = ds.get_triplets(128)
    # batch = batch_graphs(data[0])
    data = ds.get_data("training")
    # tfds = ds.build_tf_dataset("training")
    # tfds = ds.build_tf_dataset("validation")
    # gen = ds.evaluation_generator_ranking("validation")
