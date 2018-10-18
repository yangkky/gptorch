import torch
import torch.nn as nn

from gptorch import kernels, models

import numpy as np
from scipy import linalg

class FF(nn.Module):

    def __init__(self, d_in, d_out):
        super(FF, self).__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.nl = nn.Tanh()

    def forward(self, X):
        return self.nl(self.linear(X))


def test_deep():
    jitter = 1e-6
    n = 5
    m = 6
    d = 16
    d_out = 4
    X1 = np.random.random((n, d))
    X2 = np.random.random((m, d))
    y = np.random.random((n,))
    V1 = torch.tensor(X1).float()
    V2 = torch.tensor(X2).float()
    yv = torch.tensor(np.expand_dims(y, 1)).float()
    V1.requires_grad = True
    V2.requires_grad = True
    yv.requires_grad = True
    ke = kernels.SEKernel()
    network = FF(d, d_out)
    emb = network(V1)
    emb = ke(emb, emb)
    emb.sum().backward()
    mo = models.DeepGPRegressor(network, ke, prior=False)
    hi = mo.fit(V1, yv, jitter=jitter, its=100)
    print()
    sn = mo.sn.detach().numpy()[0]
    emb1 = network(V1)
    emb2 = network(V2)
    K = ke(emb1, emb1).data.numpy()
    K += (sn + jitter) * np.eye(n)
    k_star = ke(emb2, emb1).data.numpy()
    mu2 = k_star @ np.linalg.inv(K) @ np.expand_dims(y, 1)
    k_ss = ke(emb2, emb2).data.numpy()
    v2 = np.diag(k_ss - k_star @ np.linalg.inv(K) @ k_star.T)
    first = 0.5 * np.expand_dims(y, 0) @ np.linalg.inv(K) @ np.expand_dims(y, 1)
    second = 0.5 * np.log(np.linalg.det(K))
    third = 0.5 * n * np.log(2 * np.pi)
    ml = first + second
    L = np.linalg.cholesky(K)
    alpha = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, alpha, lower=False)
    alpha = np.expand_dims(alpha, 1)

    assert np.allclose(alpha, mo.alpha.detach().numpy(), atol=1e-5)
    assert np.allclose(mo.loss_func(mo.L, mo.alpha, mo.y).data.numpy(), ml)


def test_regressor():
    jitter = 1e-6
    ke = kernels.MaternKernel()
    mo = models.GPRegressor(ke, prior=False)
    print()
    n = 5
    m = 6
    d = 20
    X1 = np.random.random((n, d))
    X2 = np.random.random((m, d))
    y = np.random.random((n,))
    V1 = torch.Tensor(X1)
    V2 = torch.Tensor(X2)
    yv = torch.Tensor(np.expand_dims(y, 1))
    hi = mo.fit(V1, yv, jitter=jitter)
    print()
    sn = mo.sn.detach().numpy()[0]
    K = ke(V1, V1).data.numpy()
    K += (sn + jitter) * np.eye(n)
    k_star = ke(V2, V1).data.numpy()
    mu2 = k_star @ np.linalg.inv(K) @ np.expand_dims(y, 1)
    k_ss = ke(V2, V2).data.numpy()
    v2 = np.diag(k_ss - k_star @ np.linalg.inv(K) @ k_star.T)
    first = 0.5 * np.expand_dims(y, 0) @ np.linalg.inv(K) @ np.expand_dims(y, 1)
    second = 0.5 * np.log(np.linalg.det(K))
    third = 0.5 * n * np.log(2 * np.pi)
    ml = first + second
    L = np.linalg.cholesky(K)
    alpha = linalg.solve_triangular(L, y, lower=True)
    alpha = linalg.solve_triangular(L.T, alpha, lower=False)
    alpha = np.expand_dims(alpha, 1)

    assert np.allclose(alpha, mo.alpha.detach().numpy(), atol=1e-5)
    assert np.allclose(mo.loss_func(mo.L, mo.alpha, mo.y).data.numpy(), ml)

if __name__=="__main__":
    test_regressor()
    test_deep()
