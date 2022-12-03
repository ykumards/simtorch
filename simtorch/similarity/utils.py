import torch


def remove_negative_eigenvalues(A):
    L, V = torch.linalg.eigh(A)
    L[L < 0] = 0

    return V @ torch.diag_embed(L) @ V.T
