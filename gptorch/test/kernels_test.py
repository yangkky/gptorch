import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn

from gptorch import kernels

import numpy as np
from scipy.spatial import distance

def test_polynomial():
    s0 = np.random.random()
    sp = np.random.random()
    d = np.random.choice(np.arange(1, 5))
    ker = kernels.PolynomialKernel(int(d), s0=s0, sp=sp)
    X1 = np.random.random((3, 5))
    V1 = Variable(torch.Tensor(X1))
    X2 = np.random.random((4, 5))
    V2 = Variable(torch.Tensor(X2))
    K = (s0 ** 2 + sp ** 2 * X1 @ X2.T) ** d
    K_test = ker(V1, V2)
    assert np.allclose(K, K_test.data.numpy())

def test_cdist():
    X1 = np.random.random((3, 5))
    V1 = Variable(torch.Tensor(X1))
    X2 = np.random.random((4, 5))
    V2 = Variable(torch.Tensor(X2))

    d = distance.cdist(X1, X2)
    d_test = kernels.cdist(V1, V2)
    assert np.allclose(d, d_test.data.numpy())
    d2 = kernels.cdist(V1, V2, squared=True)
    assert np.allclose(d ** 2, d2.data.numpy())

def test_matern():
    ell = 10 * np.random.random()
    ker = kernels.MaternKernel(ell=ell)
    X1 = np.random.random((10, 5))
    V1 = Variable(torch.Tensor(X1))
    X2 = np.random.random((4, 5))
    V2 = Variable(torch.Tensor(X2))
    d = distance.cdist(X1, X2)
    D_L = d / ell
    first = (1.0 + np.sqrt(5.0) * D_L) + 5.0 * np.power(D_L, 2) / 3.0
    second = np.exp(-np.sqrt(5.0) * D_L)
    K = first * second
    K_test = ker(V1, V2)
    assert np.allclose(K, K_test.data.numpy())

def test_se():
    ell = 10 * np.random.random()
    sf = np.random.random()
    ker = kernels.SEKernel(ell=ell, sf=sf)
    X1 = np.random.random((3, 5))
    V1 = Variable(torch.Tensor(X1))
    X2 = np.random.random((4, 5))
    V2 = Variable(torch.Tensor(X2))
    d = distance.cdist(X1, X2)
    D_L = d ** 2 / ell ** 2
    K = sf ** 2 * np.exp(-0.5 * D_L)
    K_test = ker(V1, V2)
    assert np.allclose(K, K_test.data.numpy())

def naive_wdk(x1, x2, S, D, cutoff=4.5):
    subs = S[x1, x2]
    k = 0
    for i, s in enumerate(subs):
        total = 0
        for j, ss in enumerate(subs):
            if i == j:
                continue
            if D[i, j] < cutoff:
                total += ss
        k += s * total
    return k

def test_fixed_wdk():
    L = 5

    X1 = np.array([[0, 1, 2, 3, 1],
                   [0, 2, 1, 3, 2],
                   [1, 2, 2, 3, 1]])
    X2 = np.array([[1, 1, 2, 1, 0],
                   [0, 2, 1, 3, 2]])
    D = np.array([[0.0, 5.0, 3.0, 6.0, 2.0],
                  [5.0, 0.0, 5.0, 6.0, 7.0],
                  [3.0, 5.0, 0.0, 1.0, 2.0],
                  [6.0, 6.0, 1.0, 0.0, 1.0],
                  [2.0, 7.0, 2.0, 1.0, 0.0]])
    contacts = [(0, 2), (0, 4), (2, 3), (2, 4), (3, 4)]
    graph = [[2, 4, -1],
             [-1, -1, -1],
             [0, 3, 4],
             [2, 4, -1],
             [0, 2, 3]]
    S = torch.randn(size=(4, 10))
    S = S @ S.t()
    a = np.random.random()
    gamma = 1.0
    ke = kernels.FixedWDK(contacts, L, S, a=a, gamma=gamma)
    S = S.detach().numpy()

    K11 = np.zeros((len(X1), len(X1)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X1):
            K11[i, j] = naive_wdk(x1, x2, S, D)
    K22 = np.zeros((len(X2), len(X2)))
    for i, x1 in enumerate(X2):
        for j, x2 in enumerate(X2):
            K22[i, j] = naive_wdk(x1, x2, S, D)
    K12 = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K12[i, j] = naive_wdk(x1, x2, S, D)
    K1_star = np.expand_dims(np.sqrt(np.diag(K11)), 1)
    K2_star = np.expand_dims(np.sqrt(np.diag(K22)), 0)
    K12 = K12 / K1_star / K2_star
    K12 = (K12 ** gamma) * a ** 2

    K = ke(torch.tensor(X1), torch.tensor(X2)).detach().numpy()
    assert np.allclose(K12, K)


def test_wdk():
    L = 5
    X1 = np.array([[0, 1, 2, 3, 1],
                   [0, 2, 1, 3, 2],
                   [1, 2, 2, 3, 1]])
    X2 = np.array([[1, 1, 2, 1, 0],
                   [0, 2, 1, 3, 2]])
    D = np.array([[0.0, 5.0, 3.0, 6.0, 2.0],
                  [5.0, 0.0, 5.0, 6.0, 7.0],
                  [3.0, 5.0, 0.0, 1.0, 2.0],
                  [6.0, 6.0, 1.0, 0.0, 1.0],
                  [2.0, 7.0, 2.0, 1.0, 0.0]])
    contacts = [(0, 2), (0, 4), (2, 3), (2, 4), (3, 4)]
    graph = [[2, 4, -1],
             [-1, -1, -1],
             [0, 3, 4],
             [2, 4, -1],
             [0, 2, 3]]
    ke = kernels.WeightedDecompositionKernel(contacts, L, 4, 10)
    S = (ke.A @ ke.A.t()).detach().numpy()

    K11 = np.zeros((len(X1), len(X1)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X1):
            K11[i, j] = naive_wdk(x1, x2, S, D)
    K22 = np.zeros((len(X2), len(X2)))
    for i, x1 in enumerate(X2):
        for j, x2 in enumerate(X2):
            K22[i, j] = naive_wdk(x1, x2, S, D)
    K12 = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K12[i, j] = naive_wdk(x1, x2, S, D)
    K1_star = np.expand_dims(np.sqrt(np.diag(K11)), 1)
    K2_star = np.expand_dims(np.sqrt(np.diag(K22)), 0)
    K12 = K12 / K1_star / K2_star

    K = ke(torch.tensor(X1), torch.tensor(X2)).detach().numpy()
    assert np.allclose(K12, K)

def naive_swdk(x1, x2, S, w):
    k = 0
    for i, (xx1, xx2) in enumerate(zip(x1, x2)):
        s12 = S[xx1, xx2]
        others = 0
        for j, (a1, a2) in enumerate(zip(x1, x2)):
            others += S[a1, a2] * w[i, j]
        k += s12 * others
    return k

def naive_normed_swdk(x1, x2, S, w):
    k = naive_swdk(x1, x2, S, w)
    k /= np.sqrt(naive_swdk(x1, x1, S, w))
    k /= np.sqrt(naive_swdk(x2, x2, S, w))
    return k

def test_swdk():
    n1 = 4
    n2 = 5
    m = 6
    L = 10
    X1 = np.random.choice(m, size=(n1, L))
    X2 = np.random.choice(m, size=(n2, L))
    T1 = torch.LongTensor(X1)
    T2 = torch.LongTensor(X2)
    ke = kernels.SoftWeightedDecompositionKernel(L, m, 3, 2 * m,
                                                 a=1.0, gamma=1.0, dist='cos')
    K12 = ke(T1, T2).detach().numpy()
    S = ke.A @ ke.A.t()
    S = S.detach().numpy()
    w = ke.pos_emb @ ke.pos_emb.t()
    # Normalize
    norms = torch.sqrt(torch.sum(ke.pos_emb ** 2, dim=1, keepdim=True))
    w = (w / norms / norms.t() + 1) / 2
    # Set diagonal to 0
    mask = -torch.eye(w.size()[0]) + 1
    w *= mask
    w = w.detach().numpy()
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = naive_normed_swdk(X1[i], X2[j], S, w)
    assert np.allclose(K12, K)

def naive_sewdk(x1, x2, S, graph):
    k = 0
    for i, (xx1, xx2) in enumerate(zip(x1, x2)):
        s12 = S[i, xx1, xx2]
        others = 0
        for j in graph[i]:
            if j == -1:
                continue
            others += S[j, x1[j], x2[j]]
        k += s12 * others
    return k

def test_series_wdk():
    L = 5
    X1 = np.array([[0, 1, 2, 3, 1],
                   [0, 2, 1, 3, 2],
                   [1, 2, 2, 3, 1]])
    X2 = np.array([[1, 1, 2, 1, 0],
                   [0, 2, 1, 3, 2]])
    contacts = [(0, 2), (0, 4), (2, 3), (2, 4), (3, 4)]
    graph = [[2, 4, -1],
             [-1, -1, -1],
             [0, 3, 4],
             [2, 4, -1],
             [0, 2, 3]]
    ke = kernels.SeriesWDK(contacts, L, 4, 10)
    S = (ke.A @ ke.A.transpose(-1, -2)).detach().numpy()

    K11 = np.zeros((len(X1), len(X1)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X1):
            K11[i, j] = naive_sewdk(x1, x2, S, graph)
    K22 = np.zeros((len(X2), len(X2)))
    for i, x1 in enumerate(X2):
        for j, x2 in enumerate(X2):
            K22[i, j] = naive_sewdk(x1, x2, S, graph)
    K12 = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K12[i, j] = naive_sewdk(x1, x2, S, graph)
    K1_star = np.expand_dims(np.sqrt(np.diag(K11)), 1)
    K2_star = np.expand_dims(np.sqrt(np.diag(K22)), 0)
    K12 = K12 / K1_star / K2_star

    K = ke(torch.tensor(X1), torch.tensor(X2)).detach().numpy()
    assert np.allclose(K12, K)

class Embedder(nn.Module):
    def __init__(self, n_aa, dims, L):
        super(Embedder, self).__init__()
        self.emb = nn.Embedding(n_aa, dims[0])
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(dims[0] * L, dims[1])
        layers = []
        for d1, d2 in zip(dims[1:-1], dims[2:]):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(d1, d2))
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        b = len(X)
        e = self.emb(X).view(b, -1)
        e = self.lin1(self.relu(e))
        return self.layers(e)

def naive_dwdk(network, x1, x2, graph, n_aa):
    e1 = network(x1[None, :]).view(n_aa, -1)
    e2 = network(x2[None, :]).view(n_aa, -1)
    e = torch.cat([e1, e2], dim=-1)
    S = e @ e.t()
    k = 0
    for i, (xx1, xx2) in enumerate(zip(x1, x2)):
        s12 = S[xx1, xx2]
        others = 0
        for j in graph[i]:
            if j == -1:
                continue
            others += S[x1[j], x2[j]]
        k += s12 * others
    return k

def test_deep_wdk():
    L = 5
    n_aa = 4
    X1 = np.array([[0, 1, 2, 3, 1],
                   [0, 2, 1, 3, 2],
                   [1, 2, 2, 3, 1]])
    X2 = np.array([[1, 1, 2, 1, 0],
                   [0, 2, 1, 3, 2]])
    X1 = torch.tensor(X1).long()
    X2 = torch.tensor(X2).long()
    contacts = [(0, 2), (0, 4), (2, 3), (2, 4), (3, 4)]
    graph = [[2, 4, -1],
             [-1, -1, -1],
             [0, 3, 4],
             [2, 4, -1],
             [0, 2, 3]]

    embedder = Embedder(n_aa, [32, 64, 32], L)
    ke = kernels.DeepWDK(embedder, contacts, L, n_aa)

    K11 = np.zeros((len(X1), len(X1)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X1):
            K11[i, j] = naive_dwdk(embedder, x1, x2, graph, n_aa)
    K22 = np.zeros((len(X2), len(X2)))
    for i, x1 in enumerate(X2):
        for j, x2 in enumerate(X2):
            K22[i, j] = naive_dwdk(embedder, x1, x2, graph, n_aa)
    K12 = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            K12[i, j] = naive_dwdk(embedder, x1, x2, graph, n_aa)
    K1_star = np.expand_dims(np.sqrt(np.diag(K11)), 1)
    K2_star = np.expand_dims(np.sqrt(np.diag(K22)), 0)
    K12 = K12 / K1_star / K2_star
    K = ke(torch.tensor(X1), torch.tensor(X2)).detach().numpy()
    assert np.allclose(K12, K)


if __name__=="__main__":
    test_fixed_wdk()
    test_polynomial()
    test_cdist()
    test_matern()
    test_se()
    test_wdk()
    test_swdk()
    test_series_wdk()
    test_deep_wdk()
