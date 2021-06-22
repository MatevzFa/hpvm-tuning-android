import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from tuning import *
from torch import nn


from torch import nn

def plot_weights(model: nn.Module, conv_id: int) -> Figure:
    convs: "list[nn.Conv2d]" = []

    for i, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            convs.append(m)

    convs = [c for c in convs if c.kernel_size == (3,3)]

    conv = convs[conv_id]

    fig, _ = plt.subplots(4, 4)

    filters = np.array(conv.weight.data)
    for ax, filter in zip(fig.axes, filters[:len(fig.axes)]):
        ax.imshow(filter)

    return fig



def main():
    model_id = sys.argv[1]
    conv_id = int(sys.argv[2])

    info = get_model_info(model_id)

    model = info.model_factory()
    model.load_state_dict(torch.load(info.checkpoint))

    print(model)

    fig = plot_weights(model, conv_id)
    fig.set_title("2D Convolution no. {conv_id}")
    fig.set_size_inches(3, 3)
    fig.savefig(f'weights-{model_id}-conv{conv_id}.pdf', bbox_inches='tight')



if __name__ == '__main__':
    main()
