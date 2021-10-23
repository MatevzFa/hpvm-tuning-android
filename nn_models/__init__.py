import os
import sys

sys.path.append(os.getenv("HPVM_ROOT") + "/hpvm/test/dnn_benchmarks")  # noqa: do not format

from pytorch.dnn.mobilenet import *
from pytorch.dnn.mobilenet import _make_seq
from torch.nn import Module

"""e
MobileNet
"""


class MobileNetUciHar(Classifier):
    def __init__(self):
        convs = Sequential(
            _make_seq(3, 32, 3, 1),
            _make_seq(32, 64, 1, 2),
            _make_seq(64, 128, 1, 1),
            _make_seq(128, 128, 1, 2),
            _make_seq(128, 256, 1, 1),
            _make_seq(256, 256, 1, 2),
            _make_seq(256, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 2),
            _make_seq(512, 1024, 1, 1),
            *make_conv_pool_activ(1024, 1024, 1, padding=0, bias=False),
            BatchNorm2d(1024, eps=0.001),
            ReLU(),
            AvgPool2d(2)
        )
        linears = Sequential(Linear(1024, 6))
        super().__init__(convs, linears)


"""
ResNet50
"""


class Bottleneck(Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.mainline = Sequential(
            *make_conv_pool_activ(in_planes, planes, 1, stride=stride),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, planes, 3, padding=1),
            BatchNorm2d(planes, eps=0.001),
            ReLU(),
            *make_conv_pool_activ(planes, self.expansion * planes, 1),
            BatchNorm2d(self.expansion * planes, eps=0.001)
        )
        self.relu1 = ReLU()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential(
                *make_conv_pool_activ(
                    in_planes, self.expansion * planes, 1, stride=stride
                ),
                BatchNorm2d(self.expansion * planes, eps=0.001)
            )
        else:
            self.shortcut = Sequential()

    def forward(self, input_):
        return self.relu1(self.mainline(input_) + self.shortcut(input_))


class ResNet50UciHar(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(
                3, 64, 3, ReLU, pool_size=2, pool_stride=2, padding=1, stride=2
            ),
            BatchNorm2d(64, eps=0.001),
            Bottleneck(64, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 128, stride=2),
            Bottleneck(512, 128),
            Bottleneck(512, 128),
            Bottleneck(512, 256, stride=2),
        )
        linears = Sequential(Linear(4096, 6))
        super().__init__(convs, linears)
