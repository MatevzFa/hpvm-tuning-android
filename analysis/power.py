import argparse
import itertools
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
from hpvm_profiler_android import read_hpvm_configs
from matplotlib import rc
import json

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'CMU Serif'
rc('text', **{'usetex': True})


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

    #
    # Prepare x-axis data
    #
    _, confs = read_hpvm_configs(config_file)

    qos_losses = [c.qos_loss for c in confs[:len(means)]]

    data = sorted(zip(qos_losses, means, stds))

    data = data[:-1]

    qos_losses = np.array([d[0] for d in data])
    means = np.array([d[1] for d in data])
    stds = np.array([d[2] for d in data])

    return qos_losses, means, stds


def main():
    args = get_args()

    print(args)

    with open(args.plotting_json) as f:
        nns = json.load(f)

    marker = itertools.cycle(('^', 'o', 's', 'x', '.'))

    plt.figure(figsize=(5, 3))

    plt.axhline(1.0, color='gray', linestyle='--', linewidth=.5)

    for nn, config_path in nns.items():
        plt.ylabel("Energy consumption")
        plt.xlabel("QoS Loss")

        qos_losses, means, stds = get_data(args.excel_file, nn, config_path)

        plt.errorbar(
            qos_losses, means, stds,
            label="\\texttt{" + esc(nn) + "}",
            linewidth=.5, marker=next(marker), markersize=3,
            elinewidth=.5, capsize=1
        )

    plt.legend(loc='best')
    plt.savefig(args.out_fig, bbox_inches='tight')


if __name__ == '__main__':
    main()
