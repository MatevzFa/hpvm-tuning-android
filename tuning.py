import os
import shutil
import sys

sys.path.append(os.getenv("HPVM_ROOT") + "/hpvm/test/dnn_benchmarks")  # noqa: do not format

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from pytorch import dnn
from torch2hpvm import BinDataset, ModelExporter
from torch2hpvm.compile import DatasetTy
from torch.nn import Module

import nn_models


def load_datasets(data_dir: Path, data_shape: Tuple[int, int, int, int]) -> Tuple[DatasetTy, DatasetTy]:
    tuneset = BinDataset(data_dir / "tune_input.bin",
                         data_dir / "tune_labels.bin", data_shape)
    testset = BinDataset(data_dir / "test_input.bin",
                         data_dir / "test_labels.bin", data_shape)
    return tuneset, testset


def prepare_model(model: Module, checkpoint: Path) -> Module:
    model.load_state_dict(torch.load(checkpoint))
    return model


def compile_target_binary(
    *,
    model_id: str,
    model: Module,
    tuneset: DatasetTy, testset: DatasetTy,
    output_dir: Path, conf_file: Path,
    batch_size: int, max_inputs: int,
) -> Tuple[ModelExporter, Path]:

    build_dir = output_dir / "build"
    target_binary = build_dir / model_id

    exporter = ModelExporter(model, tuneset, testset,
                             output_dir, config_file=conf_file)
    exporter.generate(batch_size=batch_size, max_inputs=max_inputs).compile(
        target_binary, build_dir)

    return exporter, target_binary


def install_android_binary(
    *,
    model_id: str,
    model: Module,
    tuneset: DatasetTy, testset: DatasetTy,
    output_dir: Path, conf_file: Path,
    app_root_dir: Path,
    java_package,
    android_abi: str,
) -> Tuple[ModelExporter, Path]:

    build_dir = output_dir / "build"
    target_binary = build_dir / model_id

    exporter = ModelExporter(model, tuneset, testset,
                             output_dir, config_file=conf_file, target="android",
                             weights_prefix=f"models/{model_id}")

    exporter.generate(
        java_package=java_package).compile(target_binary, build_dir)

    app_main = Path(app_root_dir) / "app" / "src" / "main"
    app_models = app_main / "assets" / "models"
    app_params = app_models / model_id
    app_bin = app_main / "cpp" / "bin" / android_abi

    app_bin.mkdir(parents=True, exist_ok=True)
    app_models.mkdir(parents=True, exist_ok=True)

    os.remove(output_dir / "weights" / "test_input.bin")
    os.remove(output_dir / "weights" / "tune_input.bin")
    os.remove(output_dir / "weights" / "test_labels.bin")
    os.remove(output_dir / "weights" / "tune_labels.bin")

    shutil.copy(build_dir / "hpvm_c.linked.bc", app_bin)

    shutil.rmtree(app_params, ignore_errors=True)
    shutil.copytree(output_dir / "weights", app_params)

    shutil.copy(os.getenv("GLOBAL_KNOBS_PATH"), app_main / "assets")
    shutil.copy(conf_file, app_params / "confs.txt")


def compile_tuner_binary(
    *,
    model: Module,
    tuneset: DatasetTy, testset: DatasetTy,
    output_dir: Path,
    batch_size: int,
) -> Tuple[ModelExporter, Path]:

    build_dir = output_dir / "build"
    tuner_binary = build_dir / "alexnet2"

    exporter = ModelExporter(model, tuneset, testset,
                             output_dir, target="hpvm_tensor_inspect")
    exporter.generate(batch_size=batch_size).compile(tuner_binary, build_dir)

    return exporter, tuner_binary


@dataclass
class TuningArgs:
    model_id: str

    max_iter: int
    qos_tuner_threshold: int
    qos_keep_threshold: int

    batch_size: int
    output_dir: str
    model_storage_dir: str
    take_best_n: int
    cost_model: str
    qos_model: str

    out_config: Optional[str] = field(default=None)


def tuning_args() -> TuningArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_id", help="Model to tune (e.g. mobilenet_cifar10)")
    parser.add_argument("max_iter", type=int)
    parser.add_argument("qos_tuner_threshold", type=float)
    parser.add_argument("qos_keep_threshold", type=float)

    parser.add_argument("-D", "--output-dir", type=str, required=True)
    parser.add_argument("-M", "--model-storage-dir", type=str, required=True)

    parser.add_argument("-B", "--batch-size", type=int, default=500)
    parser.add_argument(
        "-C", "--out-config", type=str,
        help="Output configuration name. Used in generating configuration files and plot files"
    )

    parser.add_argument("--take-best-n", type=int, default=50)
    parser.add_argument("--cost-model", type=str, default="cost_linear")
    parser.add_argument("--qos-model", type=str, default="qos_p1")

    args = parser.parse_args()
    return TuningArgs(**vars(args))


@dataclass
class ModelInfo:
    data_dir: Path
    data_shape: Tuple[int, int, int, int]
    model_factory: Callable[[], Module]
    checkpoint: Path


_model_params_base = Path(os.getenv("MODEL_PARAMS_DIR"))

_model_infos = {
    'mobilenet_cifar10': ModelInfo(
        data_dir=_model_params_base / "mobilenet_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.MobileNet,
        checkpoint=_model_params_base / "pytorch/mobilenet_cifar10.pth.tar"
    ),
    'mobilenet_uci-har': ModelInfo(
        data_dir=_model_params_base / "mobilenet_uci-har",
        data_shape=(1450, 3, 32, 32),
        model_factory=nn_models.MobileNetUciHar,
        checkpoint=_model_params_base / "pytorch/mobilenet_uci-har.pth.tar"
    ),
    'resnet18_cifar10': ModelInfo(
        data_dir=_model_params_base / "resnet18_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.ResNet18,
        checkpoint=_model_params_base / "pytorch/resnet18_cifar10.pth.tar"
    ),
    'alexnet2_cifar10': ModelInfo(
        data_dir=_model_params_base / "alexnet2_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.AlexNet2,
        checkpoint=_model_params_base / "pytorch/alexnet2_cifar10.pth.tar"
    ),
    'vgg16_cifar10': ModelInfo(
        data_dir=_model_params_base / "vgg16_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.VGG16Cifar10,
        checkpoint=_model_params_base / "pytorch/vgg16_cifar10.pth.tar"
    ),
    'resnet50_uci-har': ModelInfo(
        data_dir=_model_params_base / "mobilenet_uci-har",
        data_shape=(1450, 3, 32, 32),
        model_factory=nn_models.ResNet50UciHar,
        checkpoint=_model_params_base / "pytorch/resnet50_uci-har.pth.tar"
    ),
}


def get_model_info(model_id: str) -> ModelInfo:
    return _model_infos[model_id]
