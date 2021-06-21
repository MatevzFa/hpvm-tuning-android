import sys

sys.path.append("/app/hpvm/test/dnn_benchmarks") # noqa

from pytorch.dnn.mobilenet import *
from pytorch.dnn.mobilenet import _make_seq


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
