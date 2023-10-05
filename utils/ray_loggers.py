import os
import json
import numpy as np
from ray.tune import logger
from ray.tune.result import (TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE)
from ray.tune.utils import flatten_dict


class JsonLogger(logger.JsonLogger):
    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]
        logger.JsonLogger.on_result(self, tmp)


# class CSVLogger(logger.CSVLogger):
#     def _init(self):
#         self.update_config(self.config)
#         logger.CSVLogger._init(self)

#     def update_config(self, config):
#         self.config = config
#         config_out = os.path.join(self.logdir, EXPR_PARAM_FILE)
#         with open(config_out, "w") as f:
#             json.dump(
#                 self.config,
#                 f,
#                 indent=2,
#                 sort_keys=True,
#                 cls=logger.SafeFallbackEncoder)


class TBXLogger(logger.TBXLogger):
    def on_result(self, result):
        step = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]

        tmp = result.copy()
        for k in [
                "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]

        flat_result = flatten_dict(tmp, delimiter="/")
        ray_prefix = ["ray", "tune"]
        valid_result = {}

        for attr, value in flat_result.items():
            if attr.startswith(("train", "val", "test", "best")):
                full_attr = attr
            else:
                full_attr = "/".join(ray_prefix + [attr])
            if (isinstance(value, tuple(logger.VALID_SUMMARY_TYPES))
                    and not np.isnan(value)):
                valid_result[full_attr] = value
                self._file_writer.add_scalar(
                    full_attr, value, global_step=step)
            elif ((isinstance(value, list) and len(value) > 0)
                    or (isinstance(value, np.ndarray) and value.size > 0)):
                raise NotImplementedError

        self.last_result = valid_result
        self._file_writer.flush()


DEFAULT_LOGGERS = (JsonLogger, TBXLogger)
