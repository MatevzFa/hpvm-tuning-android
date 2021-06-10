import shutil
from pathlib import Path

import torch
from torch2hpvm import BinDataset, ModelExporter
from torch.nn import Module

from pytorch import dnn

data_dir = Path("model_params/mobilenet_cifar10")
dataset_shape = 5000, 3, 32, 32  # NCHW format.
tuneset = BinDataset(data_dir / "tune_input.bin",
                     data_dir / "tune_labels.bin", dataset_shape)
testset = BinDataset(data_dir / "test_input.bin",
                     data_dir / "test_labels.bin", dataset_shape)


model = dnn.MobileNet6()
# checkpoint = "model_params/pytorch/mobilenet_cifar10.pth.tar"
checkpoint = "/home/matevz/coding/MAG/mobilenet-uci-har/test-6class.pth"
model.load_state_dict(torch.load(checkpoint))

output_dir = Path("./mobilenet_cifar10")
shutil.rmtree(output_dir, ignore_errors=True)
build_dir = output_dir / "build"
target_binary = build_dir / "mobilenet_cifar10"
batch_size = 50
max_inputs = batch_size * 5
conf_file = "hpvm-c/benchmarks/mobilenet_cifar10/data/tuner_confs.txt"
exporter = ModelExporter(model, tuneset, testset,
                         output_dir, config_file=conf_file, target="android")

exporter.generate(
    java_package="si.fri.matevzfa.approxhpvmdemo.ApproxHPVMWrapper").compile(target_binary, build_dir)

APP_ROOT = "/home/matevz/coding/MAG/android-approxhpvm-demo"
ANDROID_ABI = "arm64-v8a"

app_root_dir = Path(APP_ROOT)

app_bin_dir = app_root_dir / "app" / "src" / "main" / "cpp" / "bin" / ANDROID_ABI
app_models_dir = app_root_dir / "app" / "src" / "main" / "assets" / "models"
app_params_dir = app_models_dir / "mobilenet_cifar10"

app_bin_dir.mkdir(parents=True, exist_ok=True)
app_models_dir.mkdir(parents=True, exist_ok=True)
app_params_dir.mkdir(parents=True, exist_ok=True)

shutil.copy(build_dir / "hpvm_c.linked.bc", app_bin_dir)

shutil.rmtree(app_params_dir, ignore_errors=True)
shutil.copytree(output_dir / "weights", app_params_dir)

shutil.copy(conf_file, app_params_dir / "confs.txt")
