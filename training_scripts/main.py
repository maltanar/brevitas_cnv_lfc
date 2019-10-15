import argparse
import os

import torch
from brevitas.core.scaling import ScalingImplType

from trainer import Trainer

parser = argparse.ArgumentParser(description="PyTorch CIFAR10/100 Training")

# Util method to add mutually exclusive boolean
def add_bool_arg(parser, name, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true")
    group.add_argument("--no_" + name, dest=name, action="store_false")
    parser.set_defaults(**{name: default})


# Util method to pass None as a string and be recognized as None value
def none_or_str(value):
    if value == "None":
        return None
    return value

def none_or_int(value):
    if value == "None":
        return None
    return int(value)


# I/O
parser.add_argument("--datadir", help="Read config from file", default="./data/")
parser.add_argument("--dataset", help="Dataset", default="MNIST")
parser.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
parser.add_argument("--dry_run", action="store_true", help="Disable output files generation")
parser.add_argument("--log_freq", type=int, default=10)

# Execution modes
parser.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
parser.add_argument("--resume", dest="resume", type=none_or_str, help="Resume from checkpoint")
add_bool_arg(parser, "strict", default=True)
add_bool_arg(parser, "detect_nan", default=False)

# Compute resources
parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
parser.add_argument("--gpus", type=none_or_str, default="0", help="Comma separated GPUs")

# Optimizer hyperparams
parser.add_argument("--batch_size", default=128, type=int, help="batch size")
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument("--scheduler", default="STEP", type=none_or_str, help="LR Scheduler")
parser.add_argument("--milestones", type=none_or_str, default='100,150,200,250', help="Scheduler milestones")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay")
parser.add_argument("--epochs", default=500, type=int, help="Number of epochs")
parser.add_argument("--random_seed", default=123456, type=int, help="Random seed")

# Neural network Architecture
parser.add_argument("--network", default="LFC", type=none_or_str, help="neural network")
parser.add_argument("--weight_bit_width", default=None, type=none_or_int, help="Weight bit width")
parser.add_argument("--act_bit_width", default=None, type=none_or_int, help="Activation bit width")
parser.add_argument("--in_bit_width", default=None, type=none_or_int, help="Input bit width")

# Pytorch precision
torch.set_printoptions(precision=10)


class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set relative paths relative to main.py
    path_args = ["datadir", "experiments", "resume"]
    for path_arg in path_args:
        path = getattr(args, path_arg)
        if path is not None and not os.path.isabs(path):
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
            setattr(args, path_arg, abs_path)

    # Access config as an object
    config = objdict(args.__dict__)

    # Avoid creating new folders etc.
    if args.evaluate:
        args.dry_run = True

    # Init trainer
    trainer = Trainer(config)

    # Execute
    if args.evaluate:
        with torch.no_grad():
            trainer.eval_model()
    else:
        trainer.train_model()
