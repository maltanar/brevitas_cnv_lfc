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

import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d, Sequential

from .common import get_quant_conv2d, get_quant_linear, get_act_quant, get_quant_type, get_stats_op

# QuantConv2d configuration
CNV_OUT_CH_POOL = [(0, 64, False), (1, 64, True), (2, 128, False), (3, 128, True), (4, 256, False), (5, 256, False)]

# Intermediate QuantLinear configuration
INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]

# Last QuantLinear configuration
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False

# MaxPool2d configuration
POOL_SIZE = 2


class CNV(Module):

    def __init__(self, num_classes=10, weight_bit_width=None, act_bit_width=None,in_bit_width=None, in_ch=3, device="cpu"):
        super(CNV, self).__init__()
        self.device = device

        weight_quant_type = get_quant_type(weight_bit_width)
        act_quant_type = get_quant_type(act_bit_width)
        in_quant_type = get_quant_type(in_bit_width)
        stats_op = get_stats_op(weight_quant_type)

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()
        self.conv_features.append(get_act_quant(in_bit_width, in_quant_type))

        for i, out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(get_quant_conv2d(in_ch=in_ch,
                                                       out_ch=out_ch,
                                                       bit_width=weight_bit_width,
                                                       quant_type=weight_quant_type,
                                                       stats_op=stats_op))
            in_ch = out_ch
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
            if i == 5:
                self.conv_features.append(Sequential())
            self.conv_features.append(BatchNorm2d(in_ch))
            self.conv_features.append(get_act_quant(act_bit_width, act_quant_type))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(get_quant_linear(in_features=in_features,
                                                         out_features=out_features,
                                                         per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                         bit_width=weight_bit_width,
                                                         quant_type=weight_quant_type,
                                                         stats_op=stats_op))
            self.linear_features.append(BatchNorm1d(out_features))
            self.linear_features.append(get_act_quant(act_bit_width, act_quant_type))
        self.fc = get_quant_linear(in_features=LAST_FC_IN_FEATURES,
                                   out_features=num_classes,
                                   per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                   bit_width=weight_bit_width,
                                   quant_type=weight_quant_type,
                                   stats_op=stats_op)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0]).to(self.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        out = self.fc(x)
        return out
