import re
import sys
import numpy as np
import pandas as pd
from absl import flags
import argparse
import os

from collections import defaultdict
from ray.tune.logger import pretty_print
from experiment_analysis import Analysis
from ray.tune.utils import date_str

FLAGS = flags.FLAGS
flags.DEFINE_string("logdir",
                    "", "")


parser = argparse.ArgumentParser(description='Analyze model performance and compare different model parameters')
parser.add_argument('--logdir', type=str, default='./ray_results/ffmpeg/GraphSearch_ffmpeg_GRID_dml_graphsim_gnn_2022-06-29_16-52-27', help='model training log directory')
parser.add_argument('--max_iter', type=int, default=120000, help='Maximum iterations used to analyze')
parser.add_argument('--top_val', type=int, default=20, help='Top-k metrics of valuation ranking')
parser.add_argument('--top_test', type=int, default=20, help='Top-k metrics of test ranking')

args = parser.parse_args()

# graphsim + gnn
flags.DEFINE_string("logdir",
                    args.logdir, "")
flags.DEFINE_integer("max_iter", args.max_iter, "")
flags.DEFINE_string("file_type", "json", "")
flags.DEFINE_integer("top_val", args.top_val, "")
flags.DEFINE_integer("top_test", args.top_test, "")
flags.DEFINE_integer("patience", 2000, "")
flags.DEFINE_integer("mode", 2, "")
flags.DEFINE_integer("es_mode", 2, "")
FLAGS(sys.argv)

score_type_val = f"top_{FLAGS.top_val}"
score_type_test = f"top_{FLAGS.top_test}"
score_name = "(ndcg|recall)"

config_names = [
    # "lr", "n_neg_graphs", "clip_grad", "batch_size", "margin",
    "weight_decay", "weight_decay_student", "weight_decay_teacher",
    # "kd_score_T", "kd_node_T", "kd_subgraph_T",
    "kd_score_coeff", "kd_node_coeff", "kd_subgraph_coeff",
    # "kd_node_mode", "kd_subgraph_mode",
    "neighbor_enhanced", "topk_node",
    "n_subgraphs", "segment",
    # "fixed_size", "graphsim_encoder",
    "model_name"
]

analysis = Analysis(experiment_dir=FLAGS.logdir, file_type=FLAGS.file_type)
configs = analysis.get_all_configs()

results = defaultdict(list)
for path, df in analysis.trial_dataframes.items():
    config_string = "_".join([
        f"{key}={configs[path][key]}" for key in config_names])
    seed = configs[path]["seed"]
    # if seed not in [6821]:
    #     continue
    df = df.loc[df.training_iteration <= FLAGS.max_iter]
    df = df.loc[df["searcher_metric"].notna()]
    # df = df.filter(regex="val|^test|training_iteration")
    df = df.filter(
        regex="val.*(ndcg|recall|precision)|^test.*(ndcg|recall|precision)|training_iteration")
    val_metric_cols = df.filter(
        regex=f"val/scores/{score_type_val}/{score_name}")
    test_metric_cols = df.filter(
        regex=f"test/scores/{score_type_test}/{score_name}")
    df["val_metric"] = val_metric_cols.mean(axis=1)
    df["test_metric"] = test_metric_cols.mean(axis=1)
    df["seed"] = seed
    results[config_string].append((seed, df))

mean_rows = {}
for key, seeds_dfs in results.items():
    seeds, dfs = list(zip(*seeds_dfs))

    # way 1
    if FLAGS.mode == 1:
        df = pd.concat(dfs, keys=seeds)
        avg_df = df.groupby(level=1).mean()
        idx = avg_df["val_metric"].idxmax()
        best_df = pd.concat([d.loc[idx] for d in dfs], axis=1, keys=seeds)
        mean_rows[key] = best_df.mean(axis=1).to_dict()

    # way 2
    elif FLAGS.mode == 2:
        best_df = []
        df_mean = 0
        for df in dfs:
            df_mean = df_mean + df
            if FLAGS.es_mode == 1:
                idx = df["val_metric"].idxmax()
                assert np.sum(df["val_metric"] == df["val_metric"].loc[idx]) == 1
            elif FLAGS.es_mode == 2:
                best = 0.0
                wait = FLAGS.patience
                for i, value in enumerate(df.val_metric):
                    if value < best:
                        wait -= 1
                    else:
                        best = value
                        wait = FLAGS.patience
                    if wait == 0:
                        break
                idx = df.iloc[:i + 1]["val_metric"].idxmax()
            elif FLAGS.es_mode == 3:
                idx_1 = df["val_metric"].idxmax()
                idx_2 = df[f"val/scores/{score_type_val}/ndcg"].idxmax()
                idx_3 = df[f"val/scores/{score_type_val}/recall"].idxmax()
                assert np.sum(df["val_metric"] == df["val_metric"].loc[idx_1]) == 1
                assert np.sum(df[f"val/scores/{score_type_val}/ndcg"] == df[f"val/scores/{score_type_val}/ndcg"].loc[idx_2]) == 1
                assert np.sum(df[f"val/scores/{score_type_val}/recall"] == df[f"val/scores/{score_type_val}/recall"].loc[idx_3]) == 1
                print(idx_1, idx_2, idx_3)
                idx = np.max([idx_1, idx_2, idx_3])
            # assert False
            best_df.append(df.loc[idx])
        df_mean = df_mean / len(dfs)
        best_df = pd.concat(best_df, axis=1, keys=seeds)
        mean_rows[key] = best_df.mean(axis=1).to_dict()
mean_df = pd.DataFrame.from_dict(mean_rows, orient="index")
result_mean_1 = mean_df.sort_values(by="val_metric").to_dict(orient="index")
result_mean_2 = mean_df.sort_values(by="test_metric").to_dict(orient="index")

# Output
log = open(f"./log/analysis/GraphSearch_{args.max_iter}_{args.top_val}_{args.top_test}_{date_str()}.txt", mode="w", encoding="utf-8")
print("rank with valuation results:", file=log)
for one_val_para in result_mean_1:
    print(one_val_para, file=log)
    print(result_mean_1[one_val_para], file=log)
    print('\n', file=log)
print('\n\n', file=log)
print("rank with test results:", file=log)

for one_test_para in result_mean_2:
    print(one_test_para, file=log)
    print(result_mean_2[one_test_para], file=log)
    print('\n', file=log)
print('\n\n', file=log)
print("done", file=log)
log.close()

path = os.path.abspath(f"./log/analysis/GraphSearch_{args.max_iter}_{args.top_val}_{args.top_test}_{date_str()}.txt")
print("\n Analysis log is written to: ", path, "\n")
