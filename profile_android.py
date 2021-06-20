import argparse
from pathlib import Path

from hpvm_profiler_android import plot_hpvm_configs, profile_config_file

from tuning import *

from dataclasses import dataclass


@dataclass
class ProfilingArgs:
    model_id: str
    configuration: str

    batch_size: int
    output_dir: str

    max_inputs: Optional[int] = field(default=None)
    out_config: Optional[str] = field(default=None)


def profiling_args() -> ProfilingArgs:
    parser = argparse.ArgumentParser()

    parser.add_argument("model_id", help="Model to tune (e.g. mobilenet_cifar10)")
    parser.add_argument("configuration", type=str)

    parser.add_argument("-D", "--output-dir", type=str, required=True,
                        help="Directory to store artifats in")

    parser.add_argument("-B", "--batch-size", type=int, default=500)
    parser.add_argument("-M", "--max-inputs", type=int, default=None)
    parser.add_argument("-C", "--out-config", type=str,
                        help="Output configuration name. Used in generating configuration files and plot files")

    args = parser.parse_args()
    return ProfilingArgs(**vars(args))


def main():
    args = profiling_args()

    info = get_model_info(args.model_id)

    assert info.data_shape[0] % args.batch_size == 0

    tuneset, testset = load_datasets(info.data_dir, info.data_shape)

    parent = Path(".")
    conf_file = Path(args.configuration)

    target_binary, target_exporter = compile_target_binary(
        model_id=args.model_id,
        model=info.model_factory(),
        tuneset=tuneset, testset=testset,
        output_dir=Path(args.output_dir),
        conf_file=conf_file,
        batch_size=args.batch_size
    )

    out_config_file = (conf_file.stem + '.android-profiled' + conf_file.suffix)
    out_plot_path = parent / (out_config_file.stem + '.png')

    profile_config_file(target_binary, conf_file, out_config_file)
    plot_hpvm_configs(out_config_file, out_plot_path)


if __name__ == '__main__':
    main()
