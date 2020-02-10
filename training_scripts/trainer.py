# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import random
import shutil
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

from logger import *
from models.CNV import CNV
from models.LFC import LFC
from models.SFC import SFC


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Trainer(object):
    def __init__(self, config):

        # Init arguments
        self.config = config
        if config.weight_bit_width is None and config.act_bit_width is None:
            prec_name = ""
        else:
            prec_name = "_{}W{}A".format(config.weight_bit_width, config.act_bit_width)
        experiment_name = '{}{}_{}'.format(config.network, prec_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir_path = os.path.join(config.experiments, experiment_name)

        if self.config.resume:
            self.output_dir_path, _ = os.path.split(config.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

        if not config.dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            if not config.resume:
                os.mkdir(self.output_dir_path)
                os.mkdir(self.checkpoints_dir_path)
        self.logger = Logger(self.output_dir_path, config.dry_run)

        # Randomness
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

        # Datasets
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        if config.dataset == 'CIFAR10':
            train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]
            transform_train = transforms.Compose(train_transforms_list)
            builder = CIFAR10
            num_classes = 10
            in_channels = 3
        elif config.dataset == 'MNIST':
            transform_train = transform_to_tensor
            builder = MNIST
            num_classes = 10
            in_channels = 1
            pass
        else:
            raise Exception("Dataset not supported: {}".format(config.dataset))


        train_set = builder(root=config.datadir,
                            train=True,
                            download=True,
                            transform=transform_train)
        test_set = builder(root=config.datadir,
                           train=False,
                           download=True,
                           transform=transform_to_tensor)
        self.train_loader = DataLoader(train_set,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers)
        self.test_loader = DataLoader(test_set,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      num_workers=config.num_workers)

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Setup device
        if config.gpus is not None:
            config.gpus = [int(i) for i in config.gpus.split(',')]
            self.device = 'cuda:' + str(config.gpus[0])
            torch.backends.cudnn.benchmark = True
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        # Setup model
        if config.network == 'CNV':
            model = CNV(weight_bit_width=config.weight_bit_width,
                        act_bit_width=config.act_bit_width,
                        in_bit_width=config.in_bit_width,
                        num_classes=num_classes,
                        in_ch=in_channels)
        elif config.network == 'LFC':
            model = LFC(weight_bit_width=config.weight_bit_width,
                        act_bit_width=config.act_bit_width,
                        in_bit_width=config.in_bit_width,
                        num_classes=num_classes,
                        in_ch=in_channels)
        elif config.network == 'SFC':
            model = SFC(weight_bit_width=config.weight_bit_width,
                        act_bit_width=config.act_bit_width,
                        in_bit_width=config.in_bit_width,
                        num_classes=num_classes,
                        in_ch=in_channels)
        else:
            raise Exception("Model not supported")

        model = model.to(device=self.device)
        if config.gpus is not None and len(config.gpus) > 1:
            model = nn.DataParallel(model, config.gpus)
        self.model = model

        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device=self.device)

        # Resume model, if any
        if config.resume:
            print('Loading model checkpoint at: {}'.format(config.resume))
            package = torch.load(config.resume, map_location=self.device)
            model_state_dict = package['state_dict']
            self.model.load_state_dict(model_state_dict, strict=config.strict)

        # Init optimizer
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config.lr,
                                   momentum=self.config.momentum,
                                   weight_decay=self.config.weight_decay)

        # Resume optimizer, if any
        if config.resume and not config.evaluate:
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']
            if 'best_val_acc' in package.keys():
                self.best_val_acc = package['best_val_acc']
        # LR scheduler
        if config.scheduler == 'STEP':
            milestones = [int(i) for i in config.milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                         milestones=milestones,
                                         gamma=0.1)
        elif config.scheduler == 'FIXED':
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(self.config.scheduler))

        # Resume scheduler, if any
        if config.resume and not config.evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package['epoch'] - 1

    def checkpoint_best(self, epoch):
        best_path = os.path.join(self.checkpoints_dir_path, "best.tar")
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }, best_path)

    def train_model(self):

        # training starts
        if self.config.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.config.epochs):

            # Set to training mode
            self.model.train()
            self.criterion.train()

            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()


            for i, data in enumerate(self.train_loader):
                (input, target) = data
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)

                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.config.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i, len(self.train_loader))

                # training batch ends
                start_data_loading = time.time()

            # Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg > self.best_val_acc and not self.config.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch)

        # training ends
        if not self.config.dry_run:
            return os.path.join(self.checkpoints_dir_path, "best.tar")

    def eval_model(self, epoch=None):
        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data

            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            #compute loss
            loss = self.criterion(output, target)
            eval_meters.loss_time.update(time.time() - end)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            #Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg
