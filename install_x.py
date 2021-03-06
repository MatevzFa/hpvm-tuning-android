from sys import stderr


def check_env():  # noqa
    import os

    def is_set(var):
        return len(os.getenv(var, "")) > 0

    variables = [
        ("HPVM_ROOT", "Path to the root of the HPVM repository (e.g. path/to/hpvm-release)"),
        ("ANDROID_HPVM_TENSOR_RT_BUILD_DIR", "Path to the build directory where hpvm-tensor-rt-android is built."),
        (
            "GLOBAL_KNOBS_PATH",
            "Path to global_knobs.txt file. You can find it in the main HPVM repository " +
            "(e.g. $HPVM_ROOT/hpvm/projects/hpvm-tensor-rt/global_knobs.txt)"
        ),
    ]

    unset_variables = [(v, description) for v, description in variables if not is_set(v)]

    if len(unset_variables) > 0:
        print("Environment has not been setup. Check if sourced your copy of env-example.sh.", file=stderr)
        exit(-1)


check_env()  # noqa


import argparse
from dataclasses import dataclass
from pathlib import Path

from nn_models import MobileNetUciHar
from tuning import (ModelInfo, get_model_info, install_android_binary,
                    load_datasets, prepare_model)


@dataclass
class _Args:
    model_id: str
    conf_file: str
    output_dir: str
    app_root: str
    abi: str
    java_package: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_id", type=str)
    parser.add_argument("conf_file", type=str)
    parser.add_argument("-A", "--app-root", type=str, required=True)
    parser.add_argument("-J", "--java-package", type=str, required=True)
    parser.add_argument("--abi", type=str, required=True)
    parser.add_argument("-o", "--output-dir", type=str, required=True)

    return _Args(**vars(parser.parse_args()))


def main():
    args = get_args()

    info = get_model_info(args.model_id)

    tuneset, testset = load_datasets(info.data_dir, info.data_shape)

    model = prepare_model(info.model_factory(), info.checkpoint)

    install_android_binary(
        model_id=args.model_id,
        model=model,
        tuneset=tuneset, testset=testset,
        conf_file=Path(args.conf_file),
        output_dir=Path(args.output_dir),
        app_root_dir=args.app_root, android_abi=args.abi,
        java_package=args.java_package)


if __name__ == '__main__':
    main()
