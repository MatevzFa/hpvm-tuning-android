import os
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
    model: Module,
    tuneset: DatasetTy, testset: DatasetTy,
    output_dir: Path, conf_file: Path,
    batch_size: int,
) -> Tuple[ModelExporter, Path]:

    build_dir = output_dir / "build"
    target_binary = build_dir / "alexnet2"

    exporter = ModelExporter(model, tuneset, testset,
                             output_dir, config_file=conf_file)
    exporter.generate(batch_size=batch_size).compile(
        target_binary, build_dir)

    return exporter, target_binary


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


@ dataclass
class TuningArgs:
    model_id: str

    max_iter: int
    qos_tuner_threshold: int
    qos_keep_threshold: int

    batch_size: int
    output_dir: str
    model_storage_dir: str
    take_best_n: int = field(default=50)
    cost_model: str = field(default="cost_linear")
    qos_model: str = field(default="qos_p1")

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

    args = parser.parse_args()
    return TuningArgs(**vars(args))


@ dataclass
class ModelInfo:
    data_dir: Path
    data_shape: Tuple[int, int, int, int]
    model_factory: Callable[[], Module]
    checkpoint: Path


_model_params_base = Path(os.getenv("MODEL_PARAMS_DIR"))

_model_infos = {
    'mobilenet_cifar10': ModelInfo(
        data_dir=_model_params_base/"alexnet2_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.MobileNet,
        checkpoint=_model_params_base/"pytorch/alexnet2_cifar10.pth.tar"
    ),
    'resnet18_cifar10': ModelInfo(
        data_dir=_model_params_base/"resnet18_cifar10",
        data_shape=(5000, 3, 32, 32),
        model_factory=dnn.ResNet18,
        checkpoint=_model_params_base/"pytorch/resnet18_cifar10.pth.tar"
    )
}


def get_model_info(model_id: str) -> ModelInfo:
    return _model_infos[model_id]
