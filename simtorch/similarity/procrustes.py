from functools import partial
from typing import Collection, Union

import numpy as np
import torch
from tqdm import tqdm

from simtorch.model.sim_model import SimilarityModel
from simtorch.similarity.base_similarity import BaseSimilarity


class Procrustes(BaseSimilarity):
    def __init__(
        self,
        sim_model1: SimilarityModel,
        sim_model2: SimilarityModel,
        device: Union[str, torch.cuda.device] = "cpu",
    ):
        self.similarity_name = "Procrustes"
        self.device = "cpu"  # FIXME: procrustes OOMs on GPU
        self.sim_model1 = sim_model1
        self.sim_model2 = sim_model2

        self.sim_model1.model.to(self.device)
        self.sim_model2.model.to(self.device)

    def procrustes_dist(self, X: torch.tensor, Y: torch.tensor):
        frobenius_norm = partial(torch.linalg.norm, ord="fro")
        nuclear_norm = partial(torch.linalg.norm, ord="nuc")

        X = self._normalize(X)
        Y = self._normalize(Y)

        X = X / frobenius_norm(X)
        Y = Y / frobenius_norm(Y)

        return 1 - nuclear_norm(X.t() @ Y).cpu()

    def compute(self, dataloader: Collection):
        dist_matrices = []
        for X, *_ in tqdm(dataloader, total=len(dataloader)):
            X = X.to(self.device)
            N = X.shape[0]

            _ = self.sim_model1.model(X)
            _ = self.sim_model2.model(X)

            batch_dist_matrix = np.zeros((self.sim_model1.n_layers, self.sim_model2.n_layers))

            # iterate through layers
            for i, (_, activation1) in enumerate(self.sim_model1.model_activations.items()):
                for j, (_, activation2) in enumerate(self.sim_model2.model_activations.items()):
                    activation1 = activation1.reshape(N, -1)
                    activation2 = activation2.reshape(N, -1)
                    dist = self.procrustes_dist(X=activation1, Y=activation2)
                    batch_dist_matrix[i, j] = dist.item()

            dist_matrices.append(batch_dist_matrix)

        self.sim_matrix = np.zeros_like(batch_dist_matrix)
        for mat in dist_matrices:
            self.sim_matrix += mat
        self.sim_matrix /= len(batch_dist_matrix)

        return self.sim_matrix
