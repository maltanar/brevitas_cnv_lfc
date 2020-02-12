# BNN-PYNQ Brevitas experiments

This repo contains training scripts and pretrained models to recreate the LFC and CNV models
used in the [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ) repo using [Brevitas](https://github.com/Xilinx/brevitas).
These pretrained models and training scripts are courtesy of 
[Alessandro Pappalardo](https://github.com/volcacius) and [Ussama Zahid](https://github.com/ussamazahid96).

## Requirements
- Pytorch >= 1.1.0
- Brevitas (https://github.com/Xilinx/brevitas)

# Install
- Get a Pytorch >= 1.1.0 installation, e.g:
 ```bash
docker pull pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
 ```

- Within your Pytorch environment, install Brevitas preview:
 ```bash
 git clone https://github.com/Xilinx/brevitas
 cd brevitas
 pip install .
 ```

- Within your Pytorch environment, with Brevitas install, clone the training repo:
 ```bash
 git clone https://github.com/maltanar/brevitas_cnv_lfc
 ```

## Experiments

| Name     | Input quantization           | Weight quantization | Activation quantization | Brevitas Top1 | Theano Top1 |
|----------|------------------------------|---------------------|-------------------------|---------------|---------------|
| LFC_1W1A | 1 bit                        | 1 bit               | 1 bit                   | 98.88%        | 98.35%        |
| LFC_1W2A | 2 bit                        | 1 bit               | 2 bit                   | 98.99%        | 98.55%        |
| CNV_1W1A | None (original [-1,1] 8 bit) | 1 bit               | 1 bit                   | 84.22%        | 79.54%        |
| CNV_1W2A | None (original [-1,1] 8 bit) | 1 bit               | 2 bit                   | 87.80%        | 83.63%        |
| CNV_2W2A | None (original [-1,1] 8 bit) | 2 bit               | 2 bit                   | 89.03%        | 84.80%        |

## Train

A few notes on training:
- An experiments folder at */path/to/experiments* must exist before launching the training.
- Training is set to 1000 epochs for 1W1A networks, 500 otherwise. 
- Force-enabling the Pytorch JIT with the env flag PYTORCH_JIT=1 significantly speeds up training.

### LFC_1W1A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network LFC --dataset MNIST --weight_bit_width 1 --act_bit_width 1 --in_bit_width 1 --experiments /path/to/experiments
 ```

### LFC_1W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network LFC --dataset MNIST --weight_bit_width 1 --act_bit_width 2 --in_bit_width 2 --experiments /path/to/experiments
 ```

### CNV_1W1A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 1 --act_bit_width 1 --in_bit_width 8 --experiments /path/to/experiments
 ```

### CNV_1W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 1 --act_bit_width 2 --in_bit_width 8 --experiments /path/to/experiments
 ```

### CNV_2W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 2 --act_bit_width 2 --in_bit_width 8 --experiments /path/to/experiments
 ```

## Evaluate
Evaluating requires to re-specify the network configuration on the command line, as it is not loaded from the checkpoint.

### LFC_1W1A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network LFC --dataset MNIST --weight_bit_width 1 --act_bit_width 1 --in_bit_width 1 --resume /path/to/LFC_1W1A/checkpoints/best.tar --evaluate --dry_run
 ```

### LFC_1W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network LFC --dataset MNIST --weight_bit_width 1 --act_bit_width 2 --in_bit_width 2 --resume /path/to/LFC_1W2A/checkpoints/best.tar --evaluate --dry_run
 ```

### CNV_1W1A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 1 --act_bit_width 1 --in_bit_width 8 --resume /path/to/CNV_1W1A/checkpoints/best.tar --evaluate --dry_run
 ```

### CNV_1W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 1 --act_bit_width 2 --in_bit_width 8 --resume /path/to/CNV_1W2A/checkpoints/best.tar --evaluate --dry_run
 ```

### CNV_2W2A

From within the *training_scripts* folder:
 ```bash
PYTORCH_JIT=1 python main.py --network CNV --dataset CIFAR10 --weight_bit_width 2 --act_bit_width 2 --in_bit_width 8 --resume /path/to/CNV_2W2A/checkpoints/best.tar --evaluate --dry_run
 ```
