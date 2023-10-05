import sys
import csv
import argparse

from absl import flags
from absl.flags import _argument_parser
from absl.flags import _flagvalues
from absl.flags import _defines
from absl.flags import _helpers
from tabulate import tabulate

from utils.common_utils import DotDict


_helpers.disclaim_module_ids.add(id(sys.modules[__name__]))


def DEFINE_list_integer(
        name, default, help, flag_values=_flagvalues.FLAGS, **args):
    parser = ListIntegerParser()
    serializer = _argument_parser.CsvListSerializer(",")
    _defines.DEFINE(parser, name, default, help,
                    flag_values, serializer, **args)


flags.DEFINE_list_integer = DEFINE_list_integer


class ListIntegerParser(_argument_parser.ListParser):
    def parse(self, argument):
        if isinstance(argument, list):
            return argument
        elif not argument:
            return []
        elif argument == "None":
            return None
        else:
            try:
                return [int(s.strip())
                        for s in list(csv.reader([argument], strict=True))[0]]
            except csv.Error as e:
                raise ValueError("Unable to parse the value %r as a %s: %s"
                                 % (argument, self.flag_type(), e))


def flag_dict():
    _flags = flags.FLAGS.get_key_flags_for_module(
        flags.FLAGS.find_module_defining_flag("DUMMY_FLAG"))
    _flag_dict = {}
    table = []
    for flag in _flags:
        _flag_dict[flag.name] = flag.value
        table.append([flag.name, flag.value])
    print(tabulate(table,
                   headers=["Hyperparameter", "Value"], tablefmt="psql"))
    sys.stdout.flush()
    return DotDict(_flag_dict)


def flag_to_dict(flags):
    pass


def args_input():
    parser = argparse.ArgumentParser(description='model training parameters')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='GPU used, such as 0, 2'),

    parser.add_argument('--data_name', type=str, default="aids700",
                        help='Datasets: aids700, cci, code(openssl), ffmpeg')
    parser.add_argument('--model_name', type=str, default="gcn",
                        help='Model names: dml, kd, gcn, gnn, gmn, graphsim')
    parser.add_argument('--student_name', type=str, default="gcn",
                        help='Student model names: gcn, gnn')
    parser.add_argument('--teacher_name', type=str, default="gmn",
                        help='Teacher model names: gmn, graphsim')

    parser.add_argument('--kd_score_coeff', type=float, default=0.0,
                        help='Score level knowledge distillation coefficient, default 0')
    parser.add_argument('--kd_node_coeff', type=float, default=0.0,
                        help='Node level knowledge distillation coefficient, default 0')
    parser.add_argument('--kd_subgraph_coeff', type=float, default=0.0,
                        help='Subgraph level knowledge distillation coefficient, default 0')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Global weight decay, default 0')
    parser.add_argument('--weight_decay_teacher', type=float, default=1e-4,
                        help='Weight decay only for teacher model, default 1e-4')
    parser.add_argument('--weight_decay_student', type=float, default=1e-5,
                        help='Weight decay only for student model, default 1e-5')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size, default 64')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Training learning rate, default 1e-3')
    parser.add_argument('--n_training_steps', type=int, default=120000,
                        help='Training steps, default 120000')
    parser.add_argument('--eval_period', type=int, default=3000,
                        help='Evaluation period during training, default 3000')

    args = parser.parse_args()

    return args


def args2flags(args, flags_in):
    flags_in.seed = args.seed
    flags_in.gpu = args.gpu

    flags_in.data_name = args.data_name
    flags_in.model_name = args.model_name
    flags_in.student_name = args.student_name
    flags_in.teacher_name = args.teacher_name

    flags_in.kd_score_coeff = args.kd_score_coeff
    flags_in.kd_node_coeff = args.kd_node_coeff
    flags_in.kd_subgraph_coeff = args.kd_subgraph_coeff

    flags_in.weight_decay = args.weight_decay
    flags_in.weight_decay_teacher = args.weight_decay_teacher
    flags_in.weight_decay_student = args.weight_decay_student

    flags_in.batch_size = args.batch_size
    flags_in.lr = args.lr
    flags_in.n_training_steps = args.n_training_steps
    flags_in.eval_period = args.eval_period

    return flags_in
