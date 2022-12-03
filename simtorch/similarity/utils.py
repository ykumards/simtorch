import torch


def remove_negative_eigenvalues(A):
    L, V = torch.linalg.eig(A)
    L[torch.view_as_real(L)[:, 0] < 0] = 0

    return torch.view_as_real(V @ torch.diag_embed(L) @ torch.linalg.inv(V))[:, :, 0]
