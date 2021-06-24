import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import nn

from tuning import *


def plot_weights(model: nn.Module, conv_id: int) -> Figure:
    convs: "list[nn.Conv2d]" = []

    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            convs.append(m)

    convs = [c for c in convs if c.kernel_size == (3, 3)]

    conv = convs[conv_id]

    fig, axes = plt.subplots(4, 4)

    filters = np.array(conv.weight.data)
    filters = (filters - np.min(filters))/np.ptp(filters)

    for ax, filter in zip(fig.axes, filters[:len(fig.axes)]):
        c, h, w = 0, 1, 2

        f = np.transpose(filter, axes=[h, w, c])
        # f = np.sum(f, axis=2)

        ax.imshow(f)
        ax.set_frame_on(False)
        ax.tick_params(
            which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labeltop=False, labelleft=False, labelright=False,
        )

    for col, ax in enumerate(axes[0], ord('A')):
        ax.set_title(chr(col))

    for row, ax in enumerate(axes[:, 0], 1):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=10, y=.25)

    print(filters.shape)
    var = np.var(filters, axis=(1,2))

    print(var.shape)
    print(f"var min={np.min(var):.3f} avg={np.mean(var):.3f} median={np.median(var):.3f} max={np.max(var):.3f}")

    return fig


def main():
    model_id = sys.argv[1]
    conv_id = int(sys.argv[2])

    info = get_model_info(model_id)

    model = info.model_factory()
    model.load_state_dict(torch.load(info.checkpoint))

    fig = plot_weights(model, conv_id)
    # fig.suptitle(f"2D Convolution no. #{conv_id}")
    fig.set_size_inches(3, 3)
    fig.savefig(f'weights-{model_id}-conv{conv_id}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
