
import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
from pandas.core.arrays.sparse import dtype
from pandas.core.indexes.multi import MultiIndex
from pandas.core.indexes.range import RangeIndex
from sklearn import tree, linear_model, model_selection, metrics

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


def load(file):
    lines = []
    with open(file) as f:
        for l in iter_lines(f):
            if l == "begin":
                continue
            if l == "end":
                continue

            if l.startswith("conf"):
                lines.append(l.split(" "))

    df = pd.DataFrame(lines)

    return df


def process(df):

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

        for l in range(6):
            idx = (y == l)
            c_correct = avg_confidence(idx_correct & idx, y_pred, confidence)
            c_wrong = avg_confidence(idx_wrong & idx, y_pred, confidence)

            out.loc[len(out)] = [conf, acc, int(l), *c_correct, *c_wrong]

    return out


def model_tree():
    args = get_args()

    df = load(args.model_log)

    grouped = df.groupby([0])

    data = []

    for name, group in grouped:

        y = group.loc[:, 1].to_numpy(dtype=int)
        y_pred = group.loc[:, 2].to_numpy(dtype=int)
        confidence = group.loc[:, 3:].to_numpy(dtype=float)

        features = pd.DataFrame({
            'y': y,
            'y_pred': y_pred,
            'y_pred_0': (y_pred == 0),
            'y_pred_1': (y_pred == 1),
            'y_pred_2': (y_pred == 2),
            'y_pred_3': (y_pred == 3),
            'y_pred_4': (y_pred == 4),
            'y_pred_5': (y_pred == 5),
            'confidence': confidence[np.arange(len(y_pred)), y_pred],
            'confidence_var': confidence.var(axis=1),
            'correct': (y == y_pred).astype(int),
        })

        acc = features["correct"].sum() / len(features["correct"])

        train, test = model_selection.train_test_split(features, test_size=.33, shuffle=True, random_state=123)
        feats = ["y_pred_0", "y_pred_1", "y_pred_2", "y_pred_3", "y_pred_4", "y_pred_5", "confidence", "confidence_var"]
        clz = "correct"

        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
        clf.fit(train[feats], train[clz])

        preds = clf.predict_proba(test[feats])[:,1] > 0.95

        f1 = metrics.f1_score(test[clz], preds, average=None)
        p = metrics.precision_score(test[clz], preds, average=None)
        r = metrics.recall_score(test[clz], preds, average=None)
        fpr, tpr, _ = metrics.roc_curve(test[clz], clf.predict_proba(test[feats])[:, 1], pos_label=0)
        auc = metrics.auc(fpr, tpr)

        data.append([name, acc, auc, *f1, *p, *r])

    out = pd.DataFrame(data, columns=["conf", "acc", "auc", "f1_w", "f1_c", "p_w", "p_c", "r_w", "r_c"])

    base_qos = out.loc[0, "acc"]
    out["qos_loss"] = base_qos - out["acc"]
    out["acc_inv"] = 1 - out["acc"]

    out = out.sort_values("qos_loss")

    plt.figure(figsize=(5, 3))
    fields = [
        # ("f1_w", "- F1", "-"),
        ("p_w", "- P", "r-"),
        ("r_w", "- R", "r--"),
        # ("f1_c", "+ F1"),
        ("p_c", "+ P", "g-"),
        ("r_c", "+ R", "g--"),
        ("acc", "Accuracy", "k-"),
        ("acc_inv", "$1 - \\mathrm{Accuracy}$", "k--"),
        ("auc", "- AUC", "b-"),
    ]
    for f, label, line in sorted(fields, key=lambda x: str(reversed(x))):
        plt.plot(out["qos_loss"], out[f], line, label=label)
    plt.legend(loc="center left", ncol=4, bbox_to_anchor=(0, 0.65), fontsize='small')
    plt.title("Correctness prediction characteristics\nfor wrong (-) and correct (+) predictions (decision tree)")
    plt.ylabel("Precision, Recall, Accuracy")
    plt.xlabel("QoS Loss")
    plt.tight_layout()
    plt.savefig("test-tree.pdf")


def model_reg():
    args = get_args()

    df = load(args.model_log)

    grouped = df.groupby([0])

    data = []

    for name, group in grouped:

        y = group.loc[:, 1].to_numpy(dtype=int)
        y_pred = group.loc[:, 2].to_numpy(dtype=int)
        confidence = group.loc[:, 3:].to_numpy(dtype=float)

        features = pd.DataFrame({
            'y': y,
            'y_pred': y_pred,
            'y_pred_0': (y_pred == 0).astype(float),
            'y_pred_1': (y_pred == 1).astype(float),
            'y_pred_2': (y_pred == 2).astype(float),
            'y_pred_3': (y_pred == 3).astype(float),
            'y_pred_4': (y_pred == 4).astype(float),
            'y_pred_5': (y_pred == 5).astype(float),
            'confidence': confidence[np.arange(len(y_pred)), y_pred],
            'confidence_var': confidence.var(axis=1),
            'correct': (y == y_pred).astype(int),
        })

        acc = features["correct"].sum() / len(features["correct"])

        train, test = model_selection.train_test_split(features, test_size=.33, shuffle=True, random_state=123)
        feats = ["y_pred_0", "y_pred_1", "y_pred_2", "y_pred_3", "y_pred_4", "y_pred_5", "confidence", "confidence_var"]
        clz = "correct"

        clf = linear_model.LogisticRegression()
        clf.fit(train[feats], train[clz])

        preds = clf.predict(test[feats])

        f1 = metrics.f1_score(test[clz], preds, average=None)
        p = metrics.precision_score(test[clz], preds, average=None)
        r = metrics.recall_score(test[clz], preds, average=None)
        fpr, tpr, _ = metrics.roc_curve(test[clz], clf.predict_proba(test[feats])[:, 1], pos_label=0)
        auc = metrics.auc(fpr, tpr)

        data.append([name, acc, auc, *f1, *p, *r])

    out = pd.DataFrame(data, columns=["conf", "acc", "auc", "f1_w", "f1_c", "p_w", "p_c", "r_w", "r_c"])

    base_qos = out.loc[0, "acc"]
    out["qos_loss"] = base_qos - out["acc"]
    out["acc_inv"] = 1 - out["acc"]

    out = out.sort_values("qos_loss")

    plt.figure(figsize=(5, 3))
    fields = [
        # ("f1_w", "- F1", "-"),
        ("p_w", "- P", "r-"),
        ("r_w", "- R", "r--"),
        # ("f1_c", "+ F1"),
        ("p_c", "+ P", "g-"),
        ("r_c", "+ R", "g--"),
        ("acc", "Accuracy", "k-"),
        ("acc_inv", "$1 - \\mathrm{Accuracy}$", "k--"),
        ("auc", "- AUC", "b-"),
    ]
    for f, label, line in sorted(fields, key=lambda x: str(reversed(x))):
        plt.plot(out["qos_loss"], out[f], line, label=label)
    plt.legend(loc="center left", ncol=4, bbox_to_anchor=(0, 0.65), fontsize='small')
    plt.title("Correctness prediction characteristics\nfor wrong (-) and correct (+) predictions (decision tree)")
    plt.ylabel("Precision, Recall, Accuracy")
    plt.xlabel("QoS Loss")
    plt.tight_layout()
    plt.savefig("test-logreg.pdf")


def plot():
    args = get_args()

    out = process(load(args.model_log))

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
        axs[l].plot(qos_losses*100, means_w, 'r--', label="wrong")
        axs[l].plot(qos_losses*100, means_c, 'g-',  label="correct")
        axs[l].set_ylim(0.4, 1.05)
        axs[l].yaxis.set_label_position("right")
        axs[l].set_ylabel(l, rotation=0,  labelpad=8)

    fig.text(-0.05,  0.5, 'SoftMax Confidence', va='center', rotation='vertical')
    plt.xlabel(esc("QoS loss [% points]"))
    plt.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, -1.5))
    plt.savefig("test.pdf", dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    # plot()
    model_tree()
    # model_reg()
