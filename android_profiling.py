import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from hpvm_profiler_android import plot_hpvm_configs, profile_config_file
from predtuner import config_pylogger
from torch2hpvm import BinDataset, ModelExporter
from torch.nn import Module

sys.path.append(os.path.abspath(os.getenv("HPVM_ROOT") + "/hpvm/test/dnn_benchmarks"))
from pytorch import dnn  # Defined at `hpvm/test/dnn_benchmarks/pytorch/dnn`


def env(key):
    val = os.getenv(key, "")
    if len(val) == 0:
        raise RuntimeError(f"env variable {key} must be given")
    return val
    
hpvm_root = Path(env("HPVM_ROOT"))
dnn_benchmarks_root = hpvm_root / "hpvm/test/dnn_benchmarks"
model_params_base = Path(env("MODEL_PARAMS_DIR"))
global_knobs_path = Path(env("GLOBAL_KNOBS_PATH"))







def compile_binary(
    model_id: str,
    model: Module,
    dataset_shape: tuple,
    batch_size: int, max_inputs: int,
    conf_file: Path,
    output_dir: Path,
) -> Path:
    """
    Returns path to compiled binary
    """
    data_dir = model_params_base / model_id
    tuneset = BinDataset(data_dir / "tune_input.bin",
                         data_dir / "tune_labels.bin", dataset_shape)
    testset = BinDataset(data_dir / "test_input.bin",
                         data_dir / "test_labels.bin", dataset_shape)
    checkpoint = model_params_base / "pytorch" / (model_id + ".pth.tar")
    model.load_state_dict(torch.load(checkpoint))

    shutil.rmtree(output_dir, ignore_errors=True)
    build_dir = output_dir / "build"
    binary = build_dir / model_id
    exporter = ModelExporter(model, tuneset, testset,
                             output_dir, config_file=str(conf_file.name))
    exporter \
        .generate(batch_size=batch_size, max_inputs=max_inputs) \
        .compile(binary, build_dir)

    return binary


def install_via_adb(
    binary_path_host: Path,
    conf_file: Path,
    output_dir: Path,
):
    w_tmpdir = output_dir / "tmp.weights"
    subprocess.run(["adb", "push", str(binary_path_host), "/data/local/tmp"])
    subprocess.run(["adb", "push", str(conf_file), "/data/local/tmp"])
    subprocess.run(["mkdir", "-p", str(w_tmpdir)])
    subprocess.run(["cp", "-rL", str(output_dir / "weights"), str(w_tmpdir)])
    subprocess.run(
        ["adb", "push", str(w_tmpdir / "weights"), "/data/local/tmp"])
    subprocess.run(
        ["adb", "push", global_knobs_path, "/data/local/tmp"])
    subprocess.run(["rm", "-rf", str(w_tmpdir)])


def run(
    model_id: str,
    model: Module,
    dataset_shape: tuple,
    config_file: Path,
    output_dir: Path,
    batch_size=25,
    max_inputs=100,
    quiet: bool = False,
):
    out_config_file = output_dir / "hpvm_confs_profiled.txt"
    binary_path = compile_binary(
        model_id, model, dataset_shape, batch_size, max_inputs, config_file, output_dir)
    install_via_adb(
        binary_path, config_file, output_dir)
    profile_config_file(
        binary_path, config_file, out_config_file, quiet=quiet)
    plot_hpvm_configs(
        out_config_file, output_dir / "configs_profiled.png")


def run_alexnet_cifar10():
    run(
        model_id="alexnet_cifar10",
        model=dnn.AlexNet(),
        dataset_shape=(5000, 3, 32, 32),
        config_file=dnn_benchmarks_root/
            "hpvm-c/benchmarks/alexnet_cifar10/data/tuner_confs.txt",
        output_dir=Path(f"android_profiling.alexnet_cifar10"),
    )


def run_alexnet2_cifar10():
    run(
        model_id="alexnet2_cifar10",
        model=dnn.AlexNet2(),
        dataset_shape=(5000, 3, 32, 32),
        config_file=dnn_benchmarks_root/
            "hpvm-c/benchmarks/alexnet2_cifar10/data/tuner_confs.txt",
        output_dir=Path(f"android_profiling.alexnet2_cifar10"),
    )


def run_vgg16_cifar10():
    run(
        model_id="vgg16_cifar10",
        model=dnn.VGG16Cifar10(),
        dataset_shape=(5000, 3, 32, 32),
        config_file=dnn_benchmarks_root/
            "hpvm-c/benchmarks/vgg16_cifar10/data/tuner_confs.txt",
        output_dir=Path(f"android_profiling.vgg16_cifar10"),
    )


def run_mobilenet_cifar10():
    run(
        model_id="mobilenet_cifar10",
        model=dnn.MobileNet(),
        dataset_shape=(5000, 3, 32, 32),
        config_file=dnn_benchmarks_root/
            "hpvm-c/benchmarks/mobilenet_cifar10/data/tuner_confs.txt",
        output_dir=Path(f"android_profiling.mobilenet_cifar10"),
    )


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: f{sys.argv[0]} <run_model function name>")
    else:
        locals()[sys.argv[1]]()
