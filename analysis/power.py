import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from numpy.core.fromnumeric import mean

from analysis import lighten_color

PathLike = Union[Path, str]
conf_opening, conf_closing = "+++++", "-----"

@dataclass
class ConfidenceInfo:
    avg_correct: float
    avg_wrong: float
    avg_total: float


class Config:
    def __init__(
        self,
        conf_name: str,
        speedup: float,
        energy: float,
        qos: float,
        qos_loss: float,
        config_body: List[str],
        confidence_info: Optional[ConfidenceInfo] = None,
        perclass_confidence_info: Optional[Dict[int, ConfidenceInfo]] = None,
    ):
        self.conf_name = conf_name
        self.speedup = speedup
        self.energy = energy
        self.qos = qos
        self.qos_loss = qos_loss
        # We don't care about the information in this part, and we don't parse this.
        self.config_body = config_body
        self.confidence_info = confidence_info
        self.perclass_confidence_info = perclass_confidence_info

    def update_profile_results(self, speedup: float, qos: float, base_qos: float,
                               confidence_info: Optional[ConfidenceInfo] = None,
                               perclass_confidence_info: Optional[Dict[int, ConfidenceInfo]] = None):
        # recorded_base_qos = self.qos + self.qos_loss
        # if abs(recorded_base_qos - base_qos) > 0.025:
        #     raise ValueError(
        #         f"Baseline QoS mismatch. Original: {recorded_base_qos}, measured: {base_qos}"
        #     )
        self.speedup = speedup
        self.qos = qos
        self.qos_loss = base_qos - qos
        if confidence_info:
            self.confidence_info = confidence_info
        if perclass_confidence_info:
            self.perclass_confidence_info = perclass_confidence_info

    def __repr__(self) -> str:
        header_fields = [
            self.conf_name,
            self.speedup,
            self.energy,
            self.qos,
            self.qos_loss,
        ]
        if self.confidence_info:
            header_fields.extend([
                self.confidence_info.avg_total,
                self.confidence_info.avg_correct,
                self.confidence_info.avg_wrong,
            ])
        if self.perclass_confidence_info:
            for label in self.perclass_confidence_info:
                header_fields.extend([
                    self.perclass_confidence_info[label].avg_total,
                    self.perclass_confidence_info[label].avg_correct,
                    self.perclass_confidence_info[label].avg_wrong,
                ])

        header = " ".join(str(field) for field in header_fields)
        lines = [conf_opening, header, *self.config_body, conf_closing]
        return "\n".join(lines)

    __str__ = __repr__


def read_hpvm_configs(config_file: PathLike) -> Tuple[str, List[Config]]:
    # def read_hpvm_configs(config_file, config_num, temp_file):
    ret_configs = []
    with open(config_file) as f:
        text = f.read()
    # There's 1 float sitting on the first line of config file.
    # We don't use it, but want to keep that intact.
    header, *configs = text.split(conf_opening)
    header = header.strip()
    for config_text in configs:
        config_text = config_text.replace(conf_closing, "").strip()
        config_header, *config_body = config_text.splitlines()
        conf_name, *number_fields = config_header.split(" ")

        speedup, energy, qos, qos_drop = [float(s) for s in number_fields[:4]]

        confidence_info = None
        # Overall confidence
        if len(number_fields) >= 7:
            cavg_total, cavg_correct, cavg_wrong = [
                float(s) for s in number_fields[4:7]]
            confidence_info = ConfidenceInfo(
                avg_total=cavg_total,
                avg_correct=cavg_correct,
                avg_wrong=cavg_wrong,
            )

        # Per-class confidence
        perclass_confidence_info = {}
        if len(number_fields) > 7:
            the_fields = number_fields[7:]
            for label, i in enumerate(range(0, len(the_fields), 3)):
                cavg_total, cavg_correct, cavg_wrong = [
                    float(s) for s in the_fields[i:i+3]]
                perclass_confidence_info[label] = ConfidenceInfo(
                    avg_total=cavg_total,
                    avg_correct=cavg_correct,
                    avg_wrong=cavg_wrong,
                )

        ret_configs.append(
            Config(
                conf_name, speedup, energy, qos, qos_drop,
                config_body,
                confidence_info=confidence_info,
                perclass_confidence_info=perclass_confidence_info,
            )
        )
    return header, ret_configs



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
