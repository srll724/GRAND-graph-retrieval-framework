import os
import ray
import socket
import tempfile
import tensorflow as tf
from datetime import datetime
from ray import tune
from tensorflow import nest
from tensorflow.python.ops import summary_ops_v2
from fwd9m.tensorflow import enable_determinism

from utils.common_utils import flatten_dict
from utils.ray_loggers import DEFAULT_LOGGERS


class Trainable(tune.Trainable):
    def __init__(self, config=None, logger_creator=None, logdir_prefix=None):
        config = config or {}

        if logger_creator is None:
            timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            if logdir_prefix is None:
                logdir_prefix = f"{self.__class__.__name__}_{timestr}"
            result_dir = os.path.join(os.getcwd(), "ray_results")

            def default_logger_creator(config):
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                logdir = tempfile.mkdtemp(
                    prefix=logdir_prefix, dir=result_dir)
                return tune.logger.UnifiedLogger(
                    config, logdir, loggers=DEFAULT_LOGGERS)

            logger_creator = default_logger_creator

        tune.Trainable.__init__(self, config, logger_creator)

    def train_buffered(self, buffer_time_s, max_buffer_length):
        results = tune.Trainable.train_buffered(
            self, buffer_time_s, max_buffer_length)
        return results[-1:]

    def setup_tf(self, seed, gpu_ids=None):
        enable_determinism()
        cvd_str = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = tf.config.list_physical_devices("GPU")
        if cvd_str == "":
            assert len(gpus) == 0
            # assert gpu_ids is None
        elif cvd_str is None:
            if not isinstance(gpu_ids, list):
                gpu_ids = [gpu_ids]
            if gpus:
                gpus = [gpus[i] for i in gpu_ids]
        else:
            if gpus:
                assert len(list(cvd_str.split(","))) == len(gpus)

        hostname = socket.gethostname()
        if hostname.endswith("45"):  # 45号服务器
            gpu_mem_limit = 15000  # 2: 15000, 3: 9500 4: 7100
        elif hostname.endswith("38"):
            # gpu_mem_limit = 15000
            gpu_mem_limit = 15000
        elif hostname.endswith("M6"):  # 49号服务器
            gpu_mem_limit = 15000
        elif hostname.endswith("39"):
            gpu_mem_limit = 9500
        elif hostname.endswith("43"):
            gpu_mem_limit = 15000
        else:
            gpu_mem_limit = 15000
        for gpu in gpus:
            tf.config.set_visible_devices(gpu, "GPU")
            # tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental
                 .VirtualDeviceConfiguration(memory_limit=gpu_mem_limit)])

        tf.config.threading.set_intra_op_parallelism_threads(8)
        tf.config.threading.set_inter_op_parallelism_threads(8)

        tf.random.set_seed(seed)

    def legacy_setup_tf(self, seed, gpu_ids=None):
        ray_init = ray.is_initialized()
        if ray_init and gpu_ids is None:
            # we assume that each remote worker has only 1 gpu
            gpu_ids = ray.get_gpu_ids()
        elif not ray_init and gpu_ids is not None:
            # local run
            if isinstance(gpu_ids, int):
                gpu_ids = [gpu_ids]
        else:
            raise ValueError

        gpus = tf.config.list_physical_devices("GPU")
        if gpus and gpu_ids:
            for gpu in gpu_ids:
                tf.config.set_visible_devices(gpus[gpu], "GPU")
                tf.config.experimental.set_memory_growth(gpus[gpu], True)

        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

        tf.random.set_seed(seed=seed)

    def update_metrics(self, values, mode):
        assert isinstance(values, dict)
        if not hasattr(self, "metrics"):
            self.metrics = dict(train={}, val={}, test={})

        flat_values = flatten_dict(values)
        for k, v in flat_values.items():
            if k not in self.metrics[mode]:
                self.metrics[mode][k] = tf.keras.metrics.Mean(name=k)
            self.metrics[mode][k].update_state(v)

    def gather_metrics(self, mode):
        mode = self.check_mode(mode)
        results = {}
        for m in mode:
            results[m] = nest.map_structure(
                lambda x: x.result().numpy(), self.metrics[m])
        return results

    def reset_metrics(self, mode):
        mode = self.check_mode(mode)
        for m in mode:
            for metric in self.metrics[m].values():
                metric.reset_states()

    def check_mode(self, mode):
        # assert mode in ["all", "training", "validation", "test"]
        if mode == "all":
            mode = ["train", "val", "test"]
        elif isinstance(mode, str):
            mode = [mode]
        elif isinstance(mode, list):
            mode = mode
        return mode

    def write_function_graph(self, graph, name):
        writer = tf.summary.create_file_writer(os.path.join(self.logdir, name))
        with writer.as_default():
            summary_ops_v2.graph(graph)
        writer.close()
