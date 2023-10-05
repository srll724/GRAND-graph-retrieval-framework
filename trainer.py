import glob
import json
import os.path as osp
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from fwd9m.tensorflow import enable_determinism
enable_determinism()  # noqa: E402

from utils.common_utils import DotDict
from utils.trainable import Trainable
from utils.tf_utils import pairwise_euclidean_similarity
from data_loader.dataset import Dataset
from model import Model
from ranking_metrics import compute_mean as tfr_metrics
from ranking_metrics import RankingMetricKey as RMK


class Trainer(Trainable):
    def setup(self, config):
        self.FLAGS = FLAGS = DotDict(config)
        self.setup_tf(seed=FLAGS.seed, gpu_ids=FLAGS.gpu)
        if FLAGS.ablation_dele_combine == True:
            if FLAGS.kd_node_coeff == 0.1 and FLAGS.kd_subgraph_coeff == 0.1:
                raise ValueError("DONT need this coeff combine")
        # 测试训练时间
        self.total_time = 0.0

        self.ds = Dataset.instantiate(
            name=FLAGS.data_name,
            node_ordering=FLAGS.node_ordering,
            n_subgraphs=FLAGS.n_subgraphs,
            batch_size=FLAGS.batch_size,
            n_neg_graphs=FLAGS.n_neg_graphs,
            batch_size_eval=FLAGS.batch_size_eval,
            seed=FLAGS.seed)

        self.model = Model(FLAGS=FLAGS)

        # ============ initialize model weights ===================
        dummy_data = self.ds.get_data(mode="train")
        _ = self.model(dummy_data["graphs"],
                       self.ds.n_graphs_per_batch_train, training=True)
        # =========================================================

        if FLAGS.clip_grad:
            clipnorm = FLAGS.grad_clip_value
        else:
            clipnorm = None
        if FLAGS.model_name == "dml":
            self.optimizer_t = tf.keras.optimizers.Adam(
                learning_rate=FLAGS.lr, global_clipnorm=clipnorm)
            self.optimizer_s = tf.keras.optimizers.Adam(
                learning_rate=FLAGS.lr, global_clipnorm=clipnorm)
        else:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=FLAGS.lr, global_clipnorm=clipnorm)

        if FLAGS.model_name == "kd":
            ckpt_kwargs = {FLAGS.teacher_name: self.model.teacher}
        elif FLAGS.model_name == "dml":
            ckpt_kwargs = {FLAGS.model_name: self.model}
        elif FLAGS.model_name in ["gnn", "graphsim", "gcn", "gmn"]:
            ckpt_kwargs = {FLAGS.model_name: self.model.model}
        self.ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        checkpoint_dir = osp.join(self.logdir, "checkpoints")
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, checkpoint_dir, max_to_keep=150)
        if FLAGS.model_name == "kd":
            dirname = osp.expanduser(f"~/ray_results/{FLAGS.data_name}")
            if FLAGS.data_name == "cci":
                if FLAGS.teacher_name == "graphsim":
                    dirname = osp.join(dirname, "GraphSearch_cci_GRID_graphsim___2022-06-10_22-01-44")
                elif FLAGS.teacher_name == "gmn":
                    dirname = osp.join(dirname, "GraphSearch_cci_GRID_gmn___2022-06-10_22-02-00")
            elif FLAGS.data_name == "code":
                if FLAGS.teacher_name == "graphsim":
                    dirname = osp.join(dirname, "GraphSearch_code_GRID_graphsim___2022-07-18_21-01-56")
                elif FLAGS.teacher_name == "gmn":
                    dirname = osp.join(dirname, "GraphSearch_code_GRID_gmn___2022-07-18_21-04-20")
            elif FLAGS.data_name == "aids700":
                if FLAGS.teacher_name == "graphsim":
                    dirname = osp.join(dirname, "GraphSearch_aids700_GRID_graphsim___2022-06-01_21-42-28")
                elif FLAGS.teacher_name == "gmn":
                    dirname = osp.join(dirname, "GraphSearch_aids700_GRID_gmn___2022-06-05_21-24-53")
                # dirname = osp.join(dirname, "GraphSearch_aids700_GRID_graphsim-gnn-gcn-gmn___2022-01-12_14-42-18")
            elif FLAGS.data_name == "ffmpeg":
                if FLAGS.teacher_name == "graphsim":
                    dirname = osp.join(dirname, "GraphSearch_ffmpeg_GRID_graphsim___2022-06-29_16-42-10")
                elif FLAGS.teacher_name == "gmn":
                    # node 10-100
                    dirname = osp.join(dirname, "GraphSearch_ffmpeg_GRID_gmn___2022-07-18_17-29-47")
                    # node 10-200
                    dirname = osp.join(dirname, "GraphSearch_ffmpeg_GRID_gmn___2022-06-29_16-43-12")
            dirname = glob.glob(osp.join(dirname, f"*model_name={FLAGS.teacher_name}*seed={FLAGS.seed}*"))
            assert len(dirname) == 1
            dirname = dirname[0]
            with open(osp.join(dirname, "result.json"), "r") as f:
                json_list = [json.loads(line) for line in f if line]
            df = pd.json_normalize(json_list, sep="/")
            df = df[df.training_iteration <= FLAGS.n_training_steps]
            if FLAGS.data_name == "aids700":
                df["metric"] = df.filter(regex=f"val/scores/top_5/(ndcg|recall)").mean(axis=1)
            else:
                df["metric"] = df.filter(regex=f"val/scores/top_5/(ndcg|recall)").mean(axis=1)
            best = 0.0
            patience = 200
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
            assert best_iter <= FLAGS.n_training_steps
            restore_path = osp.join(dirname, "checkpoints", f"ckpt-{best_iter}")
            # dirname = osp.dirname(osp.abspath(__file__))
            # restore_path = osp.join(dirname, "tf_ckpts", FLAGS.data_name, FLAGS.teacher_name, "ckpt-120000")
            # if FLAGS.data_name == "aids700":
            #     restore_path = osp.join(dirname, "tf_ckpts/aids700/ckpt-49000")
            # elif FLAGS.data_name == "cci":
            #     restore_path = osp.join(dirname, "tf_ckpts/cci/ckpt-130000")
            # elif FLAGS.data_name == "code":
            #     if FLAGS.batch_size == 32:
            #         restore_path = osp.join(dirname, "tf_ckpts/code/ckpt-125000")
            #     elif FLAGS.batch_size == 64:
            #         restore_path = osp.join(dirname, "tf_ckpts/code/ckpt-85000")
            self.ckpt.restore(restore_path).assert_consumed()
            # self.ckpt.restore(restore_path)

        input_signature = [{"graphs": self.ds.graph_signature}]
        self.train_one_step = tf.function(
            self.train_one_step, input_signature=input_signature)

        def _forward_eval_factory(n_graphs, only_model_outputs):
            func = functools.partial(
                self.forward, n_graphs=n_graphs, training=False,
                only_model_outputs=only_model_outputs)
            return tf.function(func, input_signature=input_signature)
        self.forward_eval_cache = functools.lru_cache(
            maxsize=None)(_forward_eval_factory)

        if FLAGS.write_graph:
            self.write_function_graph(
                self.train_one_step.get_concrete_function().graph,
                name="graph_train")
            self.write_function_graph(
                self.forward_eval_cache(n_graphs=1, only_model_outputs=False)
                    .get_concrete_function().graph,
                name="graph_eval")
            import ipdb; ipdb.set_trace()

        if FLAGS.autograph:
            assert not tf.config.functions_run_eagerly()
        else:
            tf.config.run_functions_eagerly(True)
            assert tf.config.functions_run_eagerly()

    def step(self):
        train_data = self.ds.get_data(mode="train")
        train_info = self.train_one_step(train_data)

        self.update_metrics(train_info, mode="train")

        gather_modes = ["train"]

        if (self.iteration + 1) % self.FLAGS.eval_period == 0:
            self.evaluate(mode="val")
            gather_modes.append("val")

            # continue_flag = self.check_validation_results()
            if self.FLAGS.tune_mode == "GRID":
                test_flag = True
            else:
                test_flag = True
            if test_flag:
                self.evaluate(mode="test")
                gather_modes.append("test")

            # checkpoint
            if self.FLAGS.checkpoint:
                save_path = self.ckpt_manager.save(self.iteration + 1)

                # checkpoint copy
                # if self.iteration + 1 == self.best_validation_loss_iteration:
                #     copy_tf_checkpoint(save_path, "best_validation_loss")
                # if self.iteration + 1 == self.best_validation_score_iteration:
                #     copy_tf_checkpoint(save_path, f"best_validation_score-{(self.iteration + 1) // 10000}")

        results = self.gather_metrics(mode=gather_modes)
        if "val" in gather_modes:
            # if False and self.FLAGS.model_name == "dml":
            #     results["searcher_metric"] = self.best_val_metric["student"]["top_10"]
            #     for ident in ["teacher", "student"]:
            #         results[ident] = {}
            #         for k in [1, 3, 5, 10]:
            #             name = f"best_top_{k}"
            #             results[ident][name] = self.best_val_metric[ident][name[5:]]
            # else:
            #     results["searcher_metric"] = self.best_val_metric["top_10"]
            #     for k in [1, 3, 5, 10, 20]:
            #         name = f"best_top_{k}"
            #         results[name] = self.best_val_metric[name[5:]]
            if not hasattr(self, "best_val_metric"):
                self.best_val_metric = 0.0
            if self.FLAGS.model_name in ["dml", "kd", "gnn", "gcn"]:
                curr_val_metric = tf.reduce_mean([
                    self.metrics["val"][f"scores/top_20/ndcg"].result(),
                    self.metrics["val"][f"scores/top_20/recall"].result()]).numpy()
            else:
                curr_val_metric = tf.reduce_mean([
                    self.metrics["val"][f"scores/top_10/ndcg"].result(),
                    self.metrics["val"][f"scores/top_10/recall"].result()]).numpy()
            if curr_val_metric > self.best_val_metric:
                self.best_val_metric = curr_val_metric
                if "test" in gather_modes:
                    self.best_test_metrics = results["test"]
            results["searcher_metric"] = self.best_val_metric
            if "test" in gather_modes:
                results["best_test_metrics"] = self.best_test_metrics
        self.reset_metrics(mode="all")

        return results

    def forward(self, inputs, n_graphs, training,
                only_model_outputs=False, mode=None):
        outputs = self.model(
            inputs["graphs"], n_graphs=n_graphs, training=training)
        if only_model_outputs:
            return outputs
        loss, info = self.model.calculate_loss(
            outputs, inputs["graphs"], n_graphs, mode)
        return loss, info, outputs

    def train_one_step(self, inputs):
        if self.FLAGS.model_name == "dml":
            with tf.GradientTape() as tape:
                loss_t, info_t, _ = self.forward(
                    inputs, n_graphs=self.ds.n_graphs_per_batch_train,
                    training=True, mode="s->t")
            with tf.name_scope("compute_grads_t"):
                grads_t = tape.gradient(
                    loss_t, self.model.teacher.trainable_weights)
            with tf.name_scope("apply_grads_t"):
                self.optimizer_t.apply_gradients(
                    zip(grads_t, self.model.teacher.trainable_weights))

            with tf.GradientTape() as tape:
                loss_s, info_s, _ = self.forward(
                    inputs, n_graphs=self.ds.n_graphs_per_batch_train,
                    training=True, mode="t->s")
            with tf.name_scope("compute_grads_s"):
                grads_s = tape.gradient(
                    loss_s, self.model.student.trainable_weights)
            with tf.name_scope("apply_grads_s"):
                self.optimizer_s.apply_gradients(
                    zip(grads_s, self.model.student.trainable_weights))

            info = {"teacher": info_t, "student": info_s}
        else:
            with tf.GradientTape() as tape:
                loss, info, _ = self.forward(
                    inputs, n_graphs=self.ds.n_graphs_per_batch_train,
                    training=True)
            with tf.name_scope("compute_grads"):
                grads = tape.gradient(loss, self.model.trainable_weights)
            with tf.name_scope("apply_grads"):
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_weights))

        return info

    def evaluate(self, mode):
        assert mode in ["val", "test"]
        if mode == "val":
            if self.FLAGS.model_name in ["dml", "kd", "gnn", "gcn"]:
                self.evaluate_ebp("val")
            else:
                self.evaluate_mbp("val")
        elif mode == "test":
            if self.FLAGS.model_name in ["dml", "gnn", "kd", "gcn"]:
                self.evaluate_ebp("test")
            else:
                if self.FLAGS.data_name in ["aids700"]:
                    self.evaluate_mbp("test")

    @property
    def eval_metric_keys(self):
        return [RMK.NDCG, RMK.RECALL, RMK.PRECISION, RMK.MRR, RMK.MAP]

    def evaluate_mbp(self, mode):
        # assert mode == "val"
        topks = [1, 3, 5, 10, 20]
        y_pred = []
        data = getattr(self.ds, f"{mode}_mbp_data")
        for d in data:
            n_graphs = int(d.n_graphs.numpy())
            dd = {"graphs": d}
            outputs = self.forward_eval_cache(
                n_graphs=n_graphs, only_model_outputs=True)(dd)
            y_pred.append(tf.reshape(outputs["score"], [-1]))
        labels = getattr(self.ds, f"{mode}_mbp_labels")
        predictions = tf.reshape(tf.concat(y_pred, axis=0), labels.shape)
        # if self.FLAGS.data_name == "aids700":
        #     predictions = self.ds.handle_ebp_predictions(predictions, 0, mode)
        info = {}
        for k in topks:
            info[f"top_{k}"] = {
                key: tfr_metrics(key, labels, predictions, topn=k)
                for key in self.eval_metric_keys}
        self.update_metrics({"scores": info}, mode=mode)
        # if not hasattr(self, "best_val_metric"):
        #     self.best_val_metric = {}
        # for k in topks:
        #     name = f"top_{k}"
        #     if name not in self.best_val_metric:
        #         self.best_val_metric[name] = 0.0
        #     curr_val_metric = tf.reduce_mean([
        #         self.metrics["val"][f"scores/{name}/ndcg"].result(),
        #         self.metrics["val"][f"scores/{name}/recall"].result()]).numpy()
        #     if curr_val_metric > self.best_val_metric[name]:
        #         self.best_val_metric[name] = curr_val_metric

    def evaluate_ebp(self, mode):
        if self.FLAGS.model_name in ["dml", "kd"]:
            fn = self.model.student
        elif self.FLAGS.model_name in ["gnn", "gcn"]:
            fn = self.model.model

        query_data = getattr(self.ds, f"{mode}_ebp_query_data")
        candidate_data = getattr(self.ds, f"{mode}_ebp_candidate_data")
        labels = getattr(self.ds, f"{mode}_ebp_labels")

        candidate_embeddings = []
        for data in candidate_data:
            embeddings, _ = fn(data, training=False)
            candidate_embeddings.append(embeddings)
        candidate_embeddings = tf.concat(candidate_embeddings, axis=0)

        info_list = []
        weights = []
        for i, data in enumerate(query_data):
            embeddings, _ = fn(data, training=False)
            predictions = pairwise_euclidean_similarity(
                embeddings, candidate_embeddings)
            predictions = self.ds.handle_ebp_predictions(predictions, i, mode)
            this_info = {}
            if self.FLAGS.data_name == "aids700":
                ks = [1, 3, 5, 10, 20]
                ks = [5, 10, 20, 50]
            else:
                ks = [10, 20, 50, 100, 200, 300]
            for k in ks:
                this_info[f"top_{k}"] = {
                    key: tfr_metrics(key, labels[i], predictions, topn=k)
                    for key in self.eval_metric_keys}
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
        self.update_metrics({"scores": info}, mode=mode)

    def bak_evaluate(self, mode):
        assert mode in ["val", "test"]

        if self.FLAGS.model_name == "dml":
            y_pred_t, y_pred_s = [], []
        else:
            y_pred = []
        for data in self.ds.get_data(mode=f"{mode}"):
            n_graphs = int(data["graphs"].n_graphs.numpy())
            outputs = self.forward_eval_cache(
                n_graphs=n_graphs, only_model_outputs=True)(data)
            if self.FLAGS.model_name == "dml":
                y_pred_t.append(tf.reshape(outputs["score_t"], [-1]))
                y_pred_s.append(tf.reshape(outputs["score_s"], [-1]))
            else:
                y_pred.append(tf.reshape(outputs["score"], [-1]))

        labels = getattr(self.ds, f"{mode}_labels")
        if self.FLAGS.model_name == "dml":
            for ident in ["teacher", "student"]:
                if ident == "teacher":
                    y_pred = y_pred_t
                elif ident == "student":
                    y_pred = y_pred_s
                predictions = tf.reshape(
                    tf.concat(y_pred, axis=0), labels.shape)
                info = {}
                for k in [1, 3, 5, 10]:
                    MAP = tfr_metrics(RMK.MAP, labels, predictions, topn=k)
                    MRR = tfr_metrics(RMK.MRR, labels, predictions, topn=k)
                    NDCG = tfr_metrics(RMK.NDCG, labels, predictions, topn=k)
                    RECALL = tfr_metrics(RMK.RECALL, labels, predictions, topn=k)
                    info[f"top_{k}"] = {
                        "map": MAP, "mrr": MRR, "ndcg": NDCG}
                self.update_metrics({f"scores_{ident}": info}, mode=mode)
                if mode == "val":
                    if not hasattr(self, "best_val_metric"):
                        self.best_val_metric = {"teacher": {}, "student": {}}
                    for k in [1, 3, 5, 10]:
                        name = f"top_{k}"
                        if name not in self.best_val_metric[ident]:
                            self.best_val_metric[ident][name] = 0.0
                        curr_val_metric = \
                            self.metrics["val"][
                                f"scores_{ident}/top_{k}/mrr"].result() \
                            + self.metrics["val"][
                                f"scores_{ident}/top_{k}/ndcg"].result()
                        curr_val_metric = curr_val_metric.numpy() / 2
                        if curr_val_metric > self.best_val_metric[ident][name]:
                            self.best_val_metric[ident][name] = curr_val_metric
        else:
            predictions = tf.reshape(tf.concat(y_pred, axis=0), labels.shape)
            info = {}
            for k in [1, 3, 5, 10]:
                MAP = tfr_metrics(RMK.MAP, labels, predictions, topn=k)
                MRR = tfr_metrics(RMK.MRR, labels, predictions, topn=k)
                NDCG = tfr_metrics(RMK.NDCG, labels, predictions, topn=k)
                info[f"top_{k}"] = {
                    "map": MAP, "mrr": MRR, "ndcg": NDCG}
            self.update_metrics({"scores": info}, mode=mode)
            if mode == "val":
                if not hasattr(self, "best_val_metric"):
                    self.best_val_metric = {}
                for k in [1, 3, 5, 10]:
                    name = f"top_{k}"
                    if name not in self.best_val_metric:
                        self.best_val_metric[name] = 0.0
                    curr_val_metric = \
                        self.metrics["val"][f"scores/top_{k}/mrr"].result() \
                        + self.metrics["val"][f"scores/top_{k}/ndcg"].result()
                    curr_val_metric = curr_val_metric.numpy() / 2
                    if curr_val_metric > self.best_val_metric[name]:
                        self.best_val_metric[name] = curr_val_metric

    def check_validation_results(self):
        if not getattr(self.check_validation_results, "built", False):
            self.best_validation_loss = np.inf
            self.validation_loss_buffer = []
            self.best_validation_loss_iteration = 0
            self.best_validation_score = 0.0
            self.validation_score_buffer = []
            self.best_validation_score_iteration = 0
            self.check_validation_results.__func__.built = True

        # curr_validation_loss = self.metrics[
        #     "validation"]["losses/loss"].result().numpy()
        curr_validation_loss = self.metrics["validation"].get("losses/loss")
        if curr_validation_loss is None:
            curr_validation_loss = np.inf
        else:
            curr_validation_loss = curr_validation_loss.result().numpy()
        self.validation_loss_buffer.append(curr_validation_loss)
        if self.FLAGS.task == "regression":
            score_name = "scores/mse"
            sign = -1.0
        elif self.FLAGS.task == "ranking":
            score_name = "scores/overall"
            sign = 1.0
        curr_validation_score = self.metrics[
            "validation"][score_name].result().numpy() * sign
        self.validation_score_buffer.append(curr_validation_score)

        patience = self.FLAGS.patience
        latest_loss = np.mean(self.validation_loss_buffer[-patience:])
        latest_score = np.mean(self.validation_score_buffer[-patience:])

        # if (curr_validation_loss <= latest_loss
        #         or curr_validation_score >= latest_score):
        if curr_validation_score >= latest_score:
        # if curr_validation_loss <= latest_loss:
            continue_flag = True
        else:
            continue_flag = False

        if curr_validation_loss < self.best_validation_loss:
            self.best_validation_loss = curr_validation_loss
            self.best_validation_loss_iteration = self.iteration + 1
        if curr_validation_score >= self.best_validation_score:
            self.best_validation_score = curr_validation_score
            self.best_validation_score_iteration = self.iteration + 1

        info = {
            "best_validation_loss_iteration":
                self.best_validation_loss_iteration,
            "best_validation_score_iteration":
                self.best_validation_score_iteration,
            f"latest_loss_{patience}": latest_loss,
            f"latest_score_{patience}": latest_score
        }
        self.update_metrics(info, mode="validation")
        return continue_flag


def copy_tf_checkpoint(src_path, dst_name, overwrite=True):
    dirname = osp.dirname(src_path)
    # copy index
    tf.io.gfile.copy(
        f"{src_path}.index",
        osp.join(dirname, f"{dst_name}.index"),
        overwrite=overwrite)
    # copy data
    for data_file in tf.io.gfile.glob(f"{src_path}.data-?????-of-?????"):
        ext = osp.splitext(data_file)[1]
        tf.io.gfile.copy(
            data_file,
            osp.join(dirname, f"{dst_name}{ext}"),
            overwrite=overwrite)


if __name__ == "__main__":
    import os
    import time
    from ray.tune.trial import date_str
    from flags import FLAGS
    from utils.flag_utils import args2flags, args_input
    import argparse

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    args = args_input()

    FLAGS = args2flags(args, FLAGS)

    print(FLAGS.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = Trainer(
        FLAGS,
        logdir_prefix=f"Trainer_"
        f"{FLAGS.data_name}_{FLAGS.model_name}_{FLAGS.seed}_{date_str()}")

    # tf_ds = trainer.ds.build_tf_dataset()
    # ds_iter = iter(tf_ds)
    # data = next(ds_iter)
    # o = trainer.model(data, training=True)

    assert True
    total_time = 0.0
    # tf.profiler.experimental.start("profile_logdir")
    while trainer.iteration < FLAGS.n_training_steps:
        # st = time.time()
        o = trainer.train()
        # one_time = time.time() - st
        # total_time += one_time
        # if trainer.iteration == 1:
        #     total_time = total_time-one_time
        # else:
        #     print(f"{trainer.iteration}, avg_time:{total_time/(trainer.iteration-1)*1000}ms, this time:{one_time*1000}ms")
        # print(f"{trainer.iteration}, {time.time() - st}s")
    # tf.profiler.experimental.stop()
        # assert False

    # trainer.examine_tf_graph()
