import os
from typing import List

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def plot_line(x, y, path=None):
    plt.scatter(x, y, s=0.02)
    plt.show()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved to {path}")
    plt.close()


def plot_hist(x: torch.Tensor, path=None, bins=None, title="", xrange=None):
    if bins is None:
        bins = len(x) // 1000
    if xrange is None:
        xrange = x.min(), x.max()

    n, bins, patches = plt.hist(x, bins=bins)
    plt.title(title, fontsize=24)
    plt.xlim(*xrange)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved to {path}")
    plt.close()
    return n, bins, patches


# def plot_dist(x: torch.Tensor, y: torch.Tensor, bins=300, path=None, index=""):
#     x = x.detach().cpu().numpy().flatten()
#     y = y.detach().cpu().numpy().flatten()

#     def subplot(data, pos, title):
#         title = f"{title} (mean={data.mean():.2f}, std={data.std():.2f})"
#         plt.subplot(*pos)
#         plt.hist(data, bins=bins)
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)
#         # plt.legend(fontsize=16)
#         plt.title(title, fontsize=20)

#     plt.figure(figsize=(18, 3))
#     subplot(x, (1, 2, 1), title=f"GELU-{index} Input")
#     subplot(y, (1, 2, 2), title=f"GELU-{index} Output")

#     plt.tight_layout()
#     if path:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         plt.savefig(path)
#         print(f"Saved to {path}")
#     plt.close()


def plot_dist(data: List, path: str = None, title: str = "", figsize=(12, 12)):
    num_col = 2
    num_row = (len(data) - 1) // num_col + 1
    _, axes = plt.subplots(num_row, num_col, figsize=figsize)

    for i, (x, ax) in enumerate(zip(tqdm(data), axes.flat)):
        ax.hist(x.flatten(), bins=300)
        if title:
            ax.set_title(
                f"{title}-{i} output distribution\n(min={x.min().item():.3f}, max={x.max().item():.3f}, $\mu$={x.mean():.3f}, $\sigma$={x.std():.3f})",
                fontsize=14,
            )
    plt.tight_layout()

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f"Saved to {path}")
    plt.show()


def plot3d(data, figsize=(10, 7), title="", cmap="coolward", elev=30, azim=120, path=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface
    patches, channels = np.arange(data.shape[1]), np.arange(data.shape[0])
    patches, channels = np.meshgrid(patches, channels)
    surf = ax.plot_surface(patches, channels, data, cmap=cmap, edgecolor="none")

    # Set axis labels
    ax.set_xlabel("Channel")
    ax.set_ylabel("Patch")
    ax.set_zlabel("Value")
    ax.set_title(title)

    # Customize the view angle
    ax.view_init(elev=elev, azim=azim)

    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label("Value")

    # Show/save the plot
    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
    plt.show()
