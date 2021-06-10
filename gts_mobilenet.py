import shutil
from pathlib import Path

import torch
from torch2hpvm import BinDataset, ModelExporter
from torch.nn import Module

from pytorch import dnn

data_dir = Path("model_params/mobilenet_cifar10")
dataset_shape = 5000, 3, 32, 32  # NCHW format.
tuneset = BinDataset(data_dir / "tune_input.bin", data_dir / "tune_labels.bin", dataset_shape)
testset = BinDataset(data_dir / "test_input.bin", data_dir / "test_labels.bin", dataset_shape)


model = dnn.MobileNet()
checkpoint = "model_params/pytorch/mobilenet_cifar10.pth.tar"
model.load_state_dict(torch.load(checkpoint))

output_dir = Path("./mobilenet_cifar10")
shutil.rmtree(output_dir, ignore_errors=True)
build_dir = output_dir / "build"
target_binary = build_dir / "mobilenet_cifar10"
batch_size = 50
max_inputs = batch_size * 5
conf_file = "hpvm-c/benchmarks/mobilenet_cifar10/data/tuner_confs.txt"
exporter = ModelExporter(model, tuneset, testset, output_dir, config_file=conf_file)
exporter.generate(batch_size=batch_size, max_inputs=max_inputs).compile(target_binary, build_dir)

APP_ROOT = Path("/home/matevz/coding/MAG/android-example-arm-compute")
shutil.copy(build_dir / "hpvm_c.linked.bc", APP_ROOT / "app/src/main/cpp/bin/arm64-v8a")

params_dir = APP_ROOT / "app/src/main/assets/models/mobilenet_cifar10"
shutil.rmtree(params_dir, ignore_errors=True)
shutil.copytree(output_dir / "weights", params_dir)

shutil.copy(conf_file, APP_ROOT / "app/src/main/assets/models/mobilenet_cifar10" / "confs.txt")
