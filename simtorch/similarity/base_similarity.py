import os
from pathlib import Path
from typing import Collection, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class BaseSimilarity:
    def __init__(self, sim_model1, sim_model2, device="cpu", **kwargs):
        self.similarity_name = "base"

        self.sim_model1 = sim_model1
        self.sim_model2 = sim_model2
        self.device = device

        self.sim_matrix = np.zeros((3, 3))

    def _normalize(self, X):
        X = X - X.mean(0, keepdims=True)
        return X / torch.linalg.norm(X)

    def compute(self, dataloader: Collection):
        pass

    def plot_similarity(
        self,
        title: Union[None, str] = None,
        xlabel: Union[None, str] = None,
        ylabel: Union[None, str] = None,
        savefig: bool = False,
        save_path: Union[str, Path] = "../output",
        plot_cmap: str = "magma"
    ):
        sns.set(rc={"figure.dpi": 200, 'savefig.dpi': 200})
        title = (
            f"{self.similarity_name}: "
            f"{self.sim_model1.model_name} vs {self.sim_model2.model_name}"
            if title is None
            else title
        )
        xlabel = f"{self.sim_model1.model_name}" if xlabel is None else xlabel
        ylabel = f"{self.sim_model2.model_name}" if ylabel is None else ylabel

        ax = sns.heatmap(self.sim_matrix, cmap=plot_cmap, cbar=True)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if savefig:
            os.makedirs(save_path, exist_ok=True)
            filename = "_".join(title.split())
            plt.savefig(os.path.join(save_path, filename))
