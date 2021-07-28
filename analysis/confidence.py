
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.range import RangeIndex

from analysis import esc, setup_tex

setup_tex()


@dataclass
class _Args:
    model_log: str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("model_log")

    return _Args(**vars(parser.parse_args()))


def iter_lines(f):
    for l in f.readlines():
        yield l.strip()


def avg_confidence(idx, labels, confidence):
    class_confidence = confidence[np.arange(len(labels)), labels]
    filtered = class_confidence[idx]
    if len(filtered) == 0:
        return 0, 0
    return filtered.mean(), filtered.std()


def main():
    args = get_args()

    lines = []
    with open(args.model_log) as f:
        for l in iter_lines(f):
            if l == "begin":
                continue
            if l == "end":
                continue

            if l.startswith("conf"):
                lines.append(l.split(" "))

    df = pd.DataFrame(lines)

    out = pd.DataFrame(columns=[
        "conf",
        "acc",
        "class",
        "correct_mean",
        "correct_std",
        "wrong_mean",
        "wrong_std",
    ])

    for conf in df.loc[:, 0].unique():
        df_conf = df[df.loc[:, 0] == conf]
        y = df_conf.loc[:, 1].to_numpy(dtype=int)
        y_pred = df_conf.loc[:, 2].to_numpy(dtype=int)
        confidence = df_conf.loc[:, 3:].to_numpy(dtype=float)

        idx_correct = y == y_pred
        idx_wrong = y != y_pred

        acc = idx_correct.sum() / len(idx_correct)

        # print(f"correct {avg_confidence(idx_correct, y_pred, confidence)}")
        # print(f"wrong   {avg_confidence(idx_wrong, y_pred, confidence)}")

        for l in range(6):
            idx = (y == l)
            c_correct = avg_confidence(idx_correct & idx, y_pred, confidence)
            c_wrong = avg_confidence(idx_wrong & idx, y_pred, confidence)

            # print(f"label={l}")
            # print(f"  correct {c_correct[0]:.5f} {c_correct[1]:.5f}")
            # print(f"  wrong   {c_wrong[0]:.5f} {c_wrong[1]:.5f}")

            out.loc[len(out)] = [conf, acc, int(l), *c_correct, *c_wrong]

    baseline_qos, = out[out["conf"] == "conf0"]["acc"].unique()

    out["qos_loss"] = baseline_qos - out["acc"]

    out = out[out["qos_loss"] >= 0]

    out = out.sort_values("qos_loss")

    labels = out["class"].unique()
    fig, axs = plt.subplots(len(labels), 1, figsize=(3, 5), sharex=True, sharey=True)
    for l in out["class"].unique():
        means_c = out[out["class"] == l]["correct_mean"]
        means_w = out[out["class"] == l]["wrong_mean"]
        qos_losses = out[out["class"] == l]["qos_loss"]
        axs[l].plot(qos_losses*100, means_c, 'g-',  label="correct")
        axs[l].plot(qos_losses*100, means_w, 'r--', label="wrong")
        axs[l].set_ylim(0.4, 1.05)
        axs[l].yaxis.set_label_position("right")
        axs[l].set_ylabel(l, rotation=0,  labelpad=8)

    fig.text(-0.05,  0.5, 'SoftMax Confidence', va='center', rotation='vertical')
    plt.xlabel(esc("QoS loss [% points]"))
    plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -1.5))
    plt.savefig("test.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
