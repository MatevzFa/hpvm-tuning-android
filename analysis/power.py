import argparse
import itertools
import json
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hpvm_profiler_android import read_hpvm_configs
from matplotlib import rc
from numpy.core.fromnumeric import mean

from analysis import lighten_color


@dataclass
class _Args:
    excel_file: str
    plotting_json: str
    out_fig: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("excel_file")
    parser.add_argument("plotting_json")
    parser.add_argument("out_fig")

    return _Args(**vars(parser.parse_args()))


def esc(s: str):
    CHARS = {
        '&':  r'\&',
        '%':  r'\%',
        '$':  r'\$',
        '#':  r'\#',
        '_':  r'\_',
        '{':  r'\{',
        '}':  r'\}',
        '~':  r'\~',
        '^':  r'\^',
        '\\': r'\\',
    }
    return "".join([CHARS.get(c, c) for c in s])


def get_data(excel_file, sheet_name, config_file):
    df = pd.read_excel(excel_file, engine='openpyxl', sheet_name=sheet_name, header=[0, 1])
    df = df[["Batch"]].dropna(axis=0, how='all').dropna(axis=1, how='all')

    a = df["Batch"].to_numpy()

    a = a[:, 1:]

    a = a / np.mean(a[0, :])

    means = np.mean(a, axis=1)
    stds = np.std(a, axis=1)

    print(len(means))

    #
    # Prepare x-axis data
    #
    _, confs = read_hpvm_configs(config_file)
    print(len(confs))

    qos_losses = [c.qos_loss for c in confs[:len(means)]]
    speedups = [c.speedup for c in confs[:len(means)]]

    print(len(qos_losses))

    data = sorted(zip(qos_losses, means, stds, speedups))

    qos_losses = np.array([d[0] for d in data])
    means = np.array([d[1] for d in data])
    stds = np.array([d[2] for d in data])
    speedups = np.array([d[3] for d in data])

    return qos_losses, means, stds, speedups


def main():
    args = get_args()

    print(args)

    with open(args.plotting_json) as f:
        nns = json.load(f)

    markers = itertools.cycle(('^', 'o', 's', 'x', '.'))
    colors = itertools.cycle(('tab:blue', 'tab:orange', 'tab:green', 'tab:red'))

    plt.figure(figsize=(5, 3))
    plt.ylim(0.55, 1.05)

    plt.axvline(3.0, color='gray', linestyle='--', linewidth=.5)
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=.5)

    for nn, config_path in nns.items():

        qos_losses, means, stds, speedups = get_data(args.excel_file, nn, config_path)
        speedups = 1/speedups

        print(qos_losses)

        lw = .75

        m = next(markers)
        c = next(colors)

        kwargs = dict(
            color=c,
            linewidth=0,
            marker=m,
            markersize=3,
        )

        plt.errorbar(
            qos_losses, means, stds,
            # linestyle=False,
            elinewidth=0.8*lw, capsize=1, ecolor=c,
            **kwargs,
        )
        # plt.plot(
        #     qos_losses, speedups,
        #     linestyle='--',
        #     **kwargs,
        # )
        plt.plot([], [],
                 label=nn.replace("_combined", ""), **kwargs)

    # plt.plot([], [], '--', color='gray', label="Time reduction")

    plt.legend(loc='best')
    plt.ylabel("Energy consumption\n(relative to no approx.)")
    plt.xlabel("QoS Loss")
    plt.savefig(args.out_fig, bbox_inches='tight')


if __name__ == '__main__':
    main()
