import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import vit_b_32 as vit


def plot_hist(x: torch.Tensor, show=False, path=None, bins=None, title="", xrange=None):
    if bins is None:
        bins = len(x) // 3000
    if xrange is None:
        xrange = x.min(), x.max()

    x = x.detach().numpy().flatten()
    n, bins, patches = plt.hist(x, bins=bins)
    plt.title(title, fontsize=24)
    plt.xlim(*xrange)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if show:
        plt.show()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'Saved to {path}')
    plt.close()
    return n, bins, patches


gelu_activations = []
softmax_activations = []

def collect_gelu_activation(mod, iten, oten):
    gelu_activations.append(oten)

def collect_softmax_activation(mod, iten, oten):
    softmax_activations.append(oten)


model = vit()
print(model)

for name, module in model.named_modules():
    if isinstance(module, nn.GELU):
        print(f"{name=}")
        module.register_forward_hook(collect_gelu_activation)
    if isinstance(module, nn.Softmax):
        print(f"{name=}")
        module.register_forward_hook(collect_softmax_activation)


x = torch.randn(1, 3, 224, 224)
y = model(x)

rootdir = "image/vit_b_32"

print(f'{len(gelu_activations) = }')
for idx, act in enumerate(gelu_activations):
    plot_hist(act.flatten(), path=f"{rootdir}/gelu{idx}_pos.png", title=f"encoder_{idx}.mlp.gelu(+)", xrange=(0, x.max()))
    plot_hist(act.flatten(), path=f"{rootdir}/gelu{idx}_neg.png", title=f"encoder_{idx}.mlp.gelu(-)", xrange=(x.min(), 0))

# print(f'{len(softmax_activations) = }')
# for idx, act in enumerate(softmax_activations):
#     plot_hist(act.flatten(), path=f"{rootdir}/softmax{idx}.png", title=f"encoder_{idx}.msa.softmax")

