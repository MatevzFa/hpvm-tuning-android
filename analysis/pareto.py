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
    parent: Optional[str]


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("conf_file")
    parser.add_argument("--parent", type=str, default=None)

    return Args(**vars(parser.parse_args()))


def plot_configs(configs: List[Config], *args, **kwargs):
    qos_losses = [c.qos_loss for c in configs]
    speedups = [c.speedup for c in configs]
    plt.plot(qos_losses, speedups, *args, **kwargs)


def main():
    args = get_args()
    print(args)

    conf_file = Path(args.conf_file)

    _, configs = read_hpvm_configs(conf_file)
    orig_len = len(configs)

    configs = list(sorted(configs, key=lambda c: c.qos_loss))
    configs = list(filter(lambda c: c.qos_loss >= 0, configs))

    print(f"removed {orig_len - len(configs)} outright")

    pareto_front = [configs[0]]

    for c in configs[1:]:
        if c.speedup - pareto_front[-1].speedup > 1e-2:
            print(f"diff = {c.speedup - pareto_front[-1].speedup}")
            pareto_front.append(c)

    print(f"kept {len(pareto_front)} in pareto")

    pareto_file_base = Path(args.parent or conf_file.parent) / (conf_file.stem + '.pareto')

    plt.figure(figsize=(5, 2.5))
    plot_configs(pareto_front, 'b:')
    plot_configs(configs, 'r-')
    plot_configs(pareto_front, 'b^')
    plt.xlabel("QoS Loss [\% points]")
    plt.ylabel("Speedup")
    plt.savefig(str(pareto_file_base)+".pdf", bbox_inches='tight')

    for i, p in enumerate(pareto_front):
        p.conf_name = f"conf{i}"

    write_hpvm_configs("0.0", pareto_front, str(pareto_file_base)+".txt")


if __name__ == '__main__':
    setup_tex()
    main()
