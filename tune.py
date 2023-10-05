import os
import copy
import ray
import random
import numpy as np
import argparse
from ray import tune
from ray.tune.utils import deep_update, date_str
# from ray.tune.suggest.variant_generator import generate_variants
from ray.tune.schedulers import AsyncHyperBandScheduler
from ax_search import AxSearch, AxClient

from flags import FLAGS
from trainer import Trainer
from utils.ray_loggers import DEFAULT_LOGGERS

parser = argparse.ArgumentParser(description='Search parameter space to get best combination')
parser.add_argument('--gpu', type=str, default="0", help='GPU used, such as 0,2')
parser.add_argument('--num_seed', type=int, default=5, help='the number of random seed, range (1-10), default 5')

parser.add_argument('--data_name', type=str, default="aids700",
                    help='Datasets: aids700, cci, code(openssl), ffmpeg')
parser.add_argument('--model_name', type=str, default="gcn",
                    help='Model names: dml, kd, gcn, gnn, gmn, graphsim')
parser.add_argument('--student_name', type=str, default="gcn",
                    help='Student model names: gcn, gnn')
parser.add_argument('--teacher_name', type=str, default="gmn",
                    help='Teacher model names: gmn, graphsim')

parser.add_argument('--kd_score_coeff', type=float, default=[0.0], nargs="+",
                    help='Score level knowledge distillation coefficient, default 0, search example: 0.01 0.1 1.0 10')
parser.add_argument('--kd_node_coeff', type=float, default=[0.0], nargs="+",
                    help='Node level knowledge distillation coefficient, default 0, search example: 0.01 0.1 1.0 10')
parser.add_argument('--kd_subgraph_coeff', type=float, default=[0.0], nargs="+",
                    help='Subgraph level knowledge distillation coefficient, default 0, search example: 0.01 0.1 1.0 10')

parser.add_argument('--weight_decay', type=float, default=[0.0], nargs="+",
                    help='Global weight decay, default 0, search format: 1e-4 1e-5')
parser.add_argument('--weight_decay_teacher', type=float, default=[1e-4], nargs="+",
                    help='Weight decay only for teacher model, default 1e-4, search example: 1e-4 1e-5')
parser.add_argument('--weight_decay_student', type=float, default=[1e-5], nargs="+",
                    help='Weight decay only for student model, default 1e-5, search example: 1e-4 1e-5')

parser.add_argument('--batch_size', type=int, default=[32], nargs="+",
                    help='Training batch size, default 32, search example: 32 64 128')
parser.add_argument('--lr', type=float, default=[1e-3], nargs="+",
                    help='Training learning rate, default 1e-3, search example: 1e-3 1e-4')
parser.add_argument('--n_training_steps', type=int, default=120000,
                    help='Training steps, default 120000')
parser.add_argument('--eval_period', type=int, default=3000,
                    help='Evaluation period during training, default 3000')

parser.add_argument('--num_cpu', type=float, default=0.0,
                    help='Number of CPUs used per trail , default 0.0')
parser.add_argument('--num_gpu', type=float, default=0.5,
                    help='Number of GPUs used per trail, default 0.5')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TUNE_GLOBAL_CHECKPOINT_S"] = "999999"
os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
# os.environ["TUNE_MAX_LEN_IDENTIFIER"] = "220"
# os.environ["TUNE_RESULT_BUFFER_MIN_TIME_S"] = "1000"
tune.trial.MAX_LEN_IDENTIFIER = 220
tune.ray_trial_executor.TUNE_RESULT_BUFFER_MIN_TIME_S = 1000.0
tune.ray_trial_executor.TUNE_RESULT_BUFFER_LENGTH = 500

random.seed(66666)
np.random.seed(66666)


def generate_seeds(size):
    return np.random.choice(66666, size=size, replace=False)
    # return [39, 8]


num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
ray.init(num_gpus=num_gpus)

config = copy.deepcopy(FLAGS)
if FLAGS.tune_mode in ["GRID", "ASHA"]:
    if FLAGS.tune_mode == "GRID":
        search_fn = tune.grid_search
        seed = generate_seeds(size=10)[:args.num_seed]
    elif FLAGS.tune_mode == "ASHA":
        search_fn = tune.choice
        seed = generate_seeds(size=10)[:1]
    search_space = {
        # "model_name": tune.grid_search(["gmn"]),
        "data_name": args.data_name,
        "model_name": tune.grid_search([args.model_name]),
        "teacher_name": tune.grid_search([args.teacher_name]),
        "graphsim_encoder": tune.grid_search(["gnn"]),
        "student_name": tune.grid_search([args.student_name]),
        "seed": search_fn(list(seed)),
        "batch_size": search_fn(list(args.batch_size)),
        # "n_neg_graphs": search_fn([1]),
        # "clip_grad": search_fn([True]),
        "ablation_dele_combine": search_fn([False]),

        # "weight_decay": search_fn([1e-5, 1e-4]),
        # "T": grid_search([0.1, 0.3, 0.5, 0.7, 1.0, 3.0, 7.0, 10.0]),
        # "kld_coeff": grid_search([1.0, 10.0, 50.0]),

        # "kd_rank_coeff": search_fn([0.0]),
        "kd_score_T": search_fn([1.0]),
        # "kd_score_coeff": search_fn([0.5, 0.75]),
        "kd_score_coeff": search_fn(list(args.kd_score_coeff)),

        # "kd_node_mode": search_fn([1]),
        # "kd_node_T": search_fn([0.3, 0.7, 1.0, 3.0, 7.0]),
        "kd_node_coeff": search_fn(list(args.kd_node_coeff)),
        # "kd_node_coeff": search_fn([0.0]),
        "neighbor_enhanced": search_fn([False]),
        "topk_node": search_fn([False]),

        # "kd_subgraph_mode": search_fn([1]),
        # "kd_subgraph_T": search_fn([1.0]),
        "kd_subgraph_coeff": search_fn(list(args.kd_subgraph_coeff)),
        # "kd_subgraph_coeff": search_fn([0.0]),
        # "n_subgraphs": search_fn([-1]),
        # "segment": search_fn(["mean"]),

        "lr": search_fn(args.lr),
        # "graph_vec_reg_weight": search_fn([0.0]),
        "weight_decay": search_fn(list(args.weight_decay)),
        "weight_decay_teacher": search_fn(list(args.weight_decay_teacher)),
        "weight_decay_student": search_fn(list(args.weight_decay_student)),
        # "weight_decay": tune.sample_from(
        #     lambda spec: 1e-4
        #     if spec.config.model_name == "graphsim" else 1e-5),
        "eval_period": args.eval_period,
        "n_training_steps": args.n_training_steps
    }
    config = deep_update(config, search_space)
elif FLAGS.tune_mode in ["Ax+ASHA", "Ax"]:
    search_space = {
        "kd_score_T": tune.uniform(1.0, 10.0),
        "kd_score_coeff": tune.uniform(1.0, 100.0),
        # "kd_node_T": tune.uniform(1.0, 10.0),
        # "kd_node_coeff": tune.loguniform(1.0, 100.0),
        "kd_subgraph_T": tune.uniform(1.0, 10.0),
        "kd_subgraph_coeff": tune.uniform(1.0, 100.0),
        # "kd_mode": tune.choice([1, 2]),
        # "neighbor_enhanced": tune.choice([True, False])
        # "lr": tune.loguniform(1e-5, 1e-3),
        # "weight_decay": tune.loguniform(1e-5, 1e-3)
    }
    config["seed"] = generate_seeds(size=None)


# variants = list(generate_variants(config))
# assert False

if FLAGS.tune_mode == "GRID":
    scheduler = None
    searcher = None
    num_samples = 1
elif FLAGS.tune_mode == "ASHA":
    if args.data_name == "aids700":
        gp = 1200
    elif args.data_name == "cci":
        gp = 15000
    scheduler = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="searcher_metric",
        mode="max",
        max_t=config["n_training_steps"],
        # grace_period=int(R / 256),
        grace_period=gp,
        reduction_factor=3,
        # reduction_factor=2.5,
        brackets=1)
    searcher = None
    num_samples = 300
elif FLAGS.tune_mode.startswith("Ax"):
    if "ASHA" in FLAGS.tune_mode:
        scheduler = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="searcher_metric",
            mode="max",
            max_t=config["n_training_steps"],
            grace_period=80000,
            reduction_factor=2,
            brackets=1)
    ax_search_space = AxSearch.convert_search_space(search_space)
    ax_client = AxClient(random_seed=42)
    ax_client.create_experiment(
        parameters=ax_search_space,
        objective_name="searcher_metric",
        minimize=False,
        choose_generation_strategy_kwargs={
            "num_initialization_trials": num_gpus * 2,
            "max_parallelism_override": num_gpus * 2})
    searcher = AxSearch(ax_client=ax_client)
    num_samples = num_gpus * 2 * 10

if isinstance(config["model_name"], dict):
    model_name = "-".join(config["model_name"]["grid_search"])
else:
    model_name = config["model_name"]
if "dml" in model_name:
    if isinstance(config["teacher_name"], dict):
        teacher_name = "-".join(config["teacher_name"]["grid_search"])
    else:
        teacher_name = config["teacher_name"]

    if isinstance(config["student_name"], dict):
        student_name = "-".join(config["student_name"]["grid_search"])
    else:
        student_name = config["student_name"]
else:
    teacher_name = ""
    student_name = ""
tune.run(
    run_or_experiment=Trainer,
    name=f"GraphSearch_{args.data_name}_"
         f"{FLAGS.tune_mode}_{model_name}_{teacher_name}_{student_name}_{date_str()}",
    stop={"training_iteration": config["n_training_steps"]},
    scheduler=scheduler,
    search_alg=searcher,
    num_samples=num_samples,
    config=config,
    resources_per_trial={"cpu": args.num_cpu, "gpu": args.num_gpu},
    loggers=DEFAULT_LOGGERS,
    local_dir=f"~/ray_results/{args.data_name}"
)
