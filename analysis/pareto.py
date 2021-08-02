from os import name
import pathlib
from analysis import setup_tex
import argparse
from typing import List, Optional
from dataclasses import dataclass

from hpvm_profiler_android import Config, read_hpvm_configs, write_hpvm_configs
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class Args:
    conf_file: str
    out: str
    plot_only: bool


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("conf_file", nargs='+')
    parser.add_argument("--out", "-o", type=str, required=True)
    parser.add_argument("--plot-only", action='store_true')

    return Args(**vars(parser.parse_args()))


def plot_configs(configs: List[Config], *args, **kwargs):
    qos_losses = [c.qos_loss for c in configs]
    speedups = [c.speedup for c in configs]
    plt.plot(qos_losses, speedups, *args, **kwargs)


def main():
    args = get_args()

    pareto_file_base = Path(args.out)

    configs = []
    for conf_file in args.conf_file:
        conf_file = Path(conf_file)
        _, new_configs = read_hpvm_configs(conf_file)
        configs.extend(new_configs)

    orig_len = len(configs)

    plt.figure(figsize=(5.5, 3))

    if args.plot_only:

        pareto = [configs[0]]
        outliers = []

        for c in configs[1:]:
            if c.speedup >= pareto[-1].speedup:
                pareto.append(c)
            else:
                outliers.append(c)

        plot_configs(pareto, 'b.:', label="Pareto frontier")
        plot_configs(outliers, 'rx', label="Outlier")
    else:
        # Sort configurations according to QoS Loss
        configs = list(sorted(configs, key=lambda c: c.qos_loss))
        # and remove those that have negative QoS loss (insensible result)
        configs = list(filter(lambda c: c.qos_loss >= 0, configs))
        print(f"removed {orig_len - len(configs)} outright")

        # Construct a sensible pareto frontier of approximations.
        # Only accept a new configuration if it yields a speedup over the previous one
        pareto_front = [configs[0]]
        for c in configs[1:]:
            if c.speedup - pareto_front[-1].speedup > 1e-2:
                if c.qos_loss - pareto_front[-1].qos_loss > 1e-1:
                    pareto_front.append(c)
                else:
                    pareto_front[-1] = c

        print(f"kept {len(pareto_front)} in pareto")

        plot_configs(pareto_front, 'b.:', label="Pareto frontier")
        plot_configs(configs, 'r-', label="All configurations (ApproxTuner)")
        plot_configs(pareto_front, 'b.')

        for i, p in enumerate(pareto_front):
            p.conf_name = f"conf{i}"

        write_hpvm_configs("0.0", pareto_front, str(pareto_file_base)+".txt")

    plt.xlabel("QoS Loss [\% points]")
    plt.ylabel("Speedup")
    plt.legend(loc='best', fontsize='small')
    plt.savefig(str(pareto_file_base)+".pdf", bbox_inches='tight')


if __name__ == '__main__':
    setup_tex()
    main()
