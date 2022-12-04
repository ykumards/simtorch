from typing import Collection, Union

import numpy as np
import torch
from tqdm import tqdm

from simtorch.model.sim_model import SimilarityModel
from simtorch.similarity.base_similarity import BaseSimilarity


class CKA(BaseSimilarity):
    def __init__(
        self,
        sim_model1: SimilarityModel,
        sim_model2: SimilarityModel,
        device: Union[str, torch.cuda.device] = "cpu",
        unbiased: bool = False,
    ):
        self.similarity_name = "CKA"
        self.device = device
        self.sim_model1 = sim_model1
        self.sim_model2 = sim_model2
        self.unbiased = unbiased

        self.sim_model1.model.to(self.device)
        self.sim_model2.model.to(self.device)

    def _centering(self, K: torch.tensor):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, L_X: torch.tensor, L_Y: torch.tensor):
        return torch.sum(self._centering(L_X) * self._centering(L_Y))

    def linear_CKA(self, L_X: torch.tensor, L_Y: torch.tensor):
        hsic = self.linear_HSIC(L_X, L_Y)
        var1 = torch.sqrt(self.linear_HSIC(L_X, L_X))
        var2 = torch.sqrt(self.linear_HSIC(L_Y, L_Y))

        return (hsic / (var1 * var2)).detach().cpu()

    def compute(self, dataloader: Collection):
        cka_matrices = []
        for X, *_ in tqdm(dataloader, total=len(dataloader)):
            X = X.to(self.device)
            batch_size = X.shape[0]

            # forward passes to activate hooks
            _ = self.sim_model1.model(X)
            _ = self.sim_model2.model(X)

            batch_cka_matrix = np.zeros((self.sim_model1.n_layers, self.sim_model2.n_layers))

            # iterate through layers
            for i, (_, activation1) in enumerate(self.sim_model1.model_activations.items()):
                X = self._normalize(activation1.view(batch_size, -1).to(self.device))
                L_X = torch.matmul(X, X.T)

                for j, (_, activation2) in enumerate(self.sim_model2.model_activations.items()):
                    Y = self._normalize(activation2.view(batch_size, -1).to(self.device))
                    L_Y = torch.matmul(Y, Y.T)

                    layer_cka = self.linear_CKA(L_X=L_X, L_Y=L_Y)

                    batch_cka_matrix[i, j] = layer_cka.item()

            cka_matrices.append(batch_cka_matrix)

        self.sim_matrix = np.zeros_like(batch_cka_matrix)
        for mat in cka_matrices:
            self.sim_matrix += mat
        self.sim_matrix /= len(batch_cka_matrix)

        return self.sim_matrix
