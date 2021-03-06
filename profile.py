import argparse
from pathlib import Path

from hpvm_profiler import plot_hpvm_configs, profile_config_file


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("binary", type=str,
                        help="Path to the target binary binary for profiling")
    parser.add_argument("config_file", type=str,
                        help="Path to the HPVM configuration file for profiling")
    parser.add_argument("--out-dir", "-d", type=str)

    args = parser.parse_args()

    binary = Path(args.binary)
    config_file = Path(args.config_file)
    parent = Path(args.out_dir or config_file.parent)

    out_config_file = parent / \
        (config_file.stem + '.profiled' + config_file.suffix)

    plot_path = parent / (config_file.stem + '.png')
    out_plot_path = parent / (out_config_file.stem + '.png')

    profile_config_file(binary, config_file, out_config_file)
    plot_hpvm_configs(out_config_file, out_plot_path)


if __name__ == '__main__':
    main()
