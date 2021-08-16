from tuning import ModelInfo, get_model_info, install_android_binary, load_datasets, prepare_model
from nn_models import MobileNetUciHar
from dataclasses import dataclass
import argparse
from pathlib import Path


@dataclass
class _Args:
    model_id: str
    conf_file: str
    output_dir: str
    app_root: str
    abi: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_id", type=str)
    parser.add_argument("conf_file", type=str)
    parser.add_argument("-A", "--app-root", type=str, required=True)
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
    )


if __name__ == '__main__':
    main()
