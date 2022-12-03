from typing import Collection, Union

import numpy as np
import torch
from tqdm import tqdm

from simtorch.model.sim_model import SimilarityModel
from simtorch.similarity.base_similarity import BaseSimilarity
from simtorch.similarity.utils import remove_negative_eigenvalues


class DeltaCKA(BaseSimilarity):
    def __init__(
        self,
        sim_model1: SimilarityModel,
        sim_model2: SimilarityModel,
        device: Union[str, torch.cuda.device] = "cpu",
        remove_negative_eig: bool = False
    ):
        self.similarity_name = "dCKA"
        self.device = device
        self.sim_model1 = sim_model1
        self.sim_model2 = sim_model2
        self.remove_negative_eig = remove_negative_eig

        self.sim_model1.model.to(self.device)
        self.sim_model2.model.to(self.device)

    def _centering(self, K: torch.tensor):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        identity = torch.eye(n, device=self.device)
        H = identity - unit / n

        return torch.matmul(torch.matmul(H, K), H)

    def _linear_residual(self, input_y, input_dist):
        input_y = input_y.view(-1, 1)

        inv_term = torch.inverse(input_dist.T @ input_dist)
        beta = inv_term @ input_dist.T @ input_y
        y_pred = input_dist @ beta

        res_sim = (input_y - y_pred).view(-1)
        pve = (1 - res_sim.var() / input_y.var()).item()

        return res_sim, pve

    def linear_HSIC(
        self,
        X: torch.tensor,
        Y: torch.tensor,
        input_confounders: torch.tensor,
    ):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)

        input_grammian = (
            torch.matmul(input_confounders, input_confounders.T).view(-1, 1)
        )
        input_distance_1 = torch.cat(
            (torch.ones(input_grammian.shape[0], 1, device=self.device), input_grammian), 1
        )

        input_distance_2 = torch.cat(
            (torch.ones(input_grammian.shape[0], 1, device=self.device), input_grammian), 1
        )

        residuals1, pve_1 = self._linear_residual(L_X, input_distance_1)
        residuals2, pve_2 = self._linear_residual(L_Y, input_distance_2)

        residuals1 = residuals1.view(L_X.shape)
        residuals2 = residuals2.view(L_Y.shape)
        if self.remove_negative_eig:
            residuals1 = remove_negative_eigenvalues(residuals1)
            residuals2 = remove_negative_eigenvalues(residuals2)

        return (torch.sum(self._centering(residuals1) * self._centering(residuals2)), pve_1, pve_2)

    def linear_CKA(
        self,
        X: torch.tensor,
        Y: torch.tensor,
        input_confounders: torch.tensor,
    ):
        X = self._normalize(X.to(self.device))
        Y = self._normalize(Y.to(self.device))
        input_confounders = self._normalize(
            input_confounders.to(self.device))

        hsic, pve_1, pve_2 = self.linear_HSIC(X, Y, input_confounders)
        var1 = torch.sqrt(self.linear_HSIC(X, X, input_confounders)[0])
        var2 = torch.sqrt(self.linear_HSIC(Y, Y, input_confounders)[0])

        return (hsic / (var1 * var2)).detach().cpu(), pve_1, pve_2

    def compute(self, dataloader: Collection):
        cka_matrices = []
        for X, *_ in tqdm(dataloader, total=len(dataloader)):
            X = X.to(self.device)
            batch_size = X.shape[0]

            _ = self.sim_model1.model(X)
            _ = self.sim_model2.model(X)

            input_confounders = X.view(batch_size, -1)
            batch_cka_matrix = np.zeros((self.sim_model1.n_layers, self.sim_model2.n_layers))

            # iterate through layers
            for i, (_, activation1) in enumerate(self.sim_model1.model_activations.items()):
                activation1 = activation1.view(batch_size, -1)
                for j, (_, activation2) in enumerate(self.sim_model2.model_activations.items()):
                    activation2 = activation2.view(batch_size, -1)

                    layer_cka, _, _ = self.linear_CKA(
                        X=activation1,
                        Y=activation2,
                        input_confounders=input_confounders,
                    )
                    batch_cka_matrix[i, j] = layer_cka.item()

            cka_matrices.append(batch_cka_matrix)

        self.sim_matrix = np.zeros_like(batch_cka_matrix)
        for mat in cka_matrices:
            self.sim_matrix += mat
        self.sim_matrix /= len(batch_cka_matrix)
        self.sim_matrix = self.sim_matrix.T

        return self.sim_matrix
