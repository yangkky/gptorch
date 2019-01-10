import abc
import itertools

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import numpy as np


def cdist(X1, X2, squared=False):
    A = torch.sum(X1 ** 2, dim=1, keepdim=True)
    B = torch.sum(X2 ** 2, dim=1, keepdim=True)
    B = torch.t(B)
    C = 2 * torch.matmul(X1, torch.t(X2))
    D = torch.clamp(A + B - C, min=1e-12)
    if not squared:
        D = torch.sqrt(D)
    return D

class BatchedKernel(nn.Module):

    def __init__(self, kernel, batchsize=100):
        super(BatchedKernel, self).__init__()
        self.kernel = kernel
        self.batchsize = batchsize

    def forward(self, X1, X2):
        n1 = len(X1)
        n2 = len(X2)
        batches_1 = n1 // self.batchsize
        if n1 % self.batchsize != 0:
            batches_1 += 1
        batches_2 = n2 // self.batchsize
        if n2 % self.batchsize != 0:
            batches_2 += 1
        K = torch.zeros(n1, n2)
        for i in range(batches_1):
            start_1 = i * self.batchsize
            stop_1 = (i + 1) * self.batchsize
            for j in range(batches_2):
                start_2 = j * self.batchsize
                stop_2 = (j + 1) * self.batchsize
#                 batched = self.kernel(X1[start_1:stop_1], X2[start_2:stop_2])
                K[start_1:stop_1, start_2:stop_2] = checkpoint(self.kernel,
                                                               X1[start_1:stop_1],
                                                               X2[start_2:stop_2])
        return K

class BaseKernel(nn.Module):

    """ A Gaussian Process kernel.

       Attributes:
           n_hypers (int)
    """

    def __init__(self):
        """ Create a GPKernel. """
        return super(BaseKernel, self).__init__()

    def forward(self, X1, X2, hypers=None):
        """ Calculate the covariance. """
        return torch.zeros((len(X1), len(X2)))


class PolynomialKernel(BaseKernel):

    """ A Polynomial kernel of the form (s0^2 + sp^2 * x.T*x)^d

    Attributes:
    hypers (list): names of the hyperparameters required
    _deg (integer): degree of polynomial
    """

    def __init__(self, d, s0=1.0, sp=1.0):
        """ Initiate a polynomial kernel.

        Parameters:
            d (integer): degree of the polynomial
        """
        super(PolynomialKernel, self).__init__()
        if not isinstance(d, int):
            raise TypeError('d must be an integer.')
        if d < 1:
            raise ValueError('d must be greater than or equal to 1.')
        self._deg = d
        self.s0 = Parameter(torch.Tensor([s0]))
        self.sp = Parameter(torch.Tensor([sp]))


    def forward(self, X1, X2):
        """ Calculate the polynomial covariance matrix between X1 and X2.

        Parameters:
            X1: n x d
            X2: m x d

        Returns:
            K: n x m
        """
        return (self.s0 ** 2 + self.sp ** 2 * X1 @ X2.t()) ** self._deg


class MaternKernel(BaseKernel):

    """ A Matern kernel with nu = 5/2. """

    def __init__(self, ell=1.0, a=1.0):
        """ Initiate a Matern kernel.  """
        super(MaternKernel, self).__init__()
        self.ell = Parameter(torch.Tensor([ell]))
        self.a = Parameter(torch.Tensor([a]))


    def forward(self, X1, X2, hypers=(1.0, )):
        """ Calculate the Matern kernel between X1 and X2.

        Parameters:
            X1: n x d
            X2: m x d

        Returns:
            K: n x m
        """
        D = cdist(X1, X2)
        D_L = D / self.ell
        sqrt5 = Variable(torch.sqrt(torch.Tensor([5.0])))
        first = (1.0 + sqrt5 * D_L) + 5.0 * D_L ** 2 / 3.0
        second = torch.exp(-sqrt5 * D_L)
        return self.a * first * second



class SEKernel(BaseKernel):

    """ A squared exponential kernel.

    Attributes:
        hypers (list)
        _d_squared (np.ndarray)
        _saved_X (dict)
    """

    def __init__(self, sf=1.0, ell=1.0):
        """ Initiate a SEKernel. """
        super(SEKernel, self).__init__()
        self.ell = Parameter(torch.Tensor([ell]))
        self.sf = Parameter(torch.Tensor([sf]))

    def forward(self, X1, X2):
        """ Calculate the squared exponential kernel between x1 and x2.

        Parameters:
            X1: n x d
            X2: m x d

        Returns:
            K: n x m
        """
        D = cdist(X1, X2) ** 2
        D_L2 = D / self.ell ** 2
        return self.sf ** 2 * torch.exp(-0.5 * D_L2)


class WeightedDecompositionKernel(BaseKernel):

    def __init__(self, contacts, L, n_S, d, a=1.0):
        super(WeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        A = torch.empty(n_S, d)
        nn.init.normal_(A, 0, 1)
        self.A = Parameter(A)
        self.graph = self.make_graph(contacts, L)
        self.graph.to(self.A.device)
        self.n_S = n_S

    def make_graph(self, contacts, L):
        graph = [[] for i in range(L)]
        for c1, c2 in contacts:
            graph[c1].append(int(c2))
            graph[c2].append(int(c1))
        max_L = max([len(g) for g in graph])
        # Fill with -1s so that every row has the same length
        graph = [g + [-1] * (max_L - len(g)) for g in graph]
        return torch.LongTensor(graph)

    def wdk(self, subs):
        n = len(subs)
        subs = torch.cat([subs, torch.zeros((n, 1))], dim=1)
        return torch.sum(subs[:, self.graph].sum(dim=2) * subs[:, :-1], dim=1)

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()
        S = self.A @ self.A.t()
        subs = S[X1, X1]
        k1 = self.wdk(subs).view((n1, 1))
        subs = S[X2, X2]
        k2 = self.wdk(subs).unsqueeze(0)
        L_inds = torch.arange(L).long()
        subs = S[X1][:, L_inds, X2].view((n1 * n2, L))
        K = self.wdk(subs).view((n1, n2))
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K


class SoftWeightedDecompositionKernel(BaseKernel):
    """ Pairwise attention, one shared substitution matrix"""

    def __init__(self, L, n_S, sub_dim, a=1.0):
        super(SoftWeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        A = torch.empty(n_S, sub_dim)
        nn.init.normal_(A, 0, 1)
        self.A = Parameter(A)
        self.n_S = n_S
        self.w = Parameter(torch.zeros(L * (L - 1) // 2))
        self.sig = nn.Sigmoid()
        i_x, i_y = np.tril_indices(L, k=-1)
        self.i_x = torch.tensor(i_x).long()
        self.i_y = torch.tensor(i_y).long()
        self.graph = torch.LongTensor([list(range(L)) for i in range(L)])
        self.graph = self.graph.to(self.A.device)

    def wdk(self, subs, w):
        temp = subs[:, self.graph] * w
        temp = temp.sum(dim=2)
        return torch.sum(temp * subs, dim=1)

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()
        w = torch.zeros(L, L)
        w[self.i_x, self.i_y] = self.w
        w[self.i_y, self.i_x] = self.w
        w = self.sig(w)
        S = self.A @ self.A.t()
        subs = S[X1, X1]
        k1 = self.wdk(subs, w).view((n1, 1))
        subs = S[X2, X2]
        k2 = self.wdk(subs, w).unsqueeze(0)
        L_inds = torch.arange(L).long()
        subs = S[X1][:, L_inds, X2].view((n1 * n2, L))
        K = self.wdk(subs, w).view((n1, n2))
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K


class DeepWDK(SoftWeightedDecompositionKernel):
    """Pairwise attention, one S for each input sequence"""

    def __init__(self, network, n_aa, L, a=1.0):
        super(SoftWeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        self.network = network
        self.n_aa = n_aa
        self.w = Parameter(torch.zeros(L * (L - 1) // 2))
        self.sig = nn.Sigmoid()
        i_x, i_y = np.tril_indices(L, k=-1)
        self.i_x = torch.tensor(i_x).long()
        self.i_y = torch.tensor(i_y).long()
        self.graph = torch.LongTensor([list(range(L)) for i in range(L)])
        self.graph = self.graph.to(self.w.device)

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()

        w = torch.zeros(L, L)
        w[self.i_x, self.i_y] = self.w
        w[self.i_y, self.i_x] = self.w
        w = self.sig(w)

        S1 = self.network(X1).view(n1, self.n_aa, -1)
        S1 = torch.bmm(S1, S1.transpose(-1, -2))
        S2 = self.network(X2).view(n2, self.n_aa, -1)
        S2 = torch.bmm(S2, S2.transpose(-1, -2))
        idx = X1 * (self.n_aa + 1)
        idx += (torch.arange(n1).long() * self.n_aa ** 2).view(n1, 1)
        subs = S1.view(-1)[idx.view(-1)].view(n1, L)
        k1 = self.wdk(subs, w).view((n1, 1))
        idx = X2 * (self.n_aa + 1)
        idx += (torch.arange(n2).long() * self.n_aa ** 2).view(n2, 1)
        subs = S2.view(-1)[idx.view(-1)].view(n2, L)
        k2 = self.wdk(subs, w).unsqueeze(0)
        inds = torch.arange(n1).long()[:, None]
        inds = inds.expand(n1, n2).contiguous().view(-1)
        S = torch.index_select(S1, 0, inds)
        inds = torch.arange(n2).long()
        inds = inds.repeat(n1)
        S += torch.index_select(S2, 0, inds)
        S /= 2
        X1m = X1.repeat(1, n2).view(n1 * n2, L)
        X2m = X2.repeat(n1, 1)
        idx = X2m + X1m * (self.n_aa)
        idx += (torch.arange(n1 * n2).long() * self.n_aa ** 2).view(n1 * n2, 1)
        subs = S.view(-1)[idx.view(-1)].view(n1 * n2, L)
        K = self.wdk(subs, w).view((n1, n2))
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K

class DeepSeriesWDK(DeepWDK):
    """Hard contacts, L substitution matrices per input"""

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()

        w = torch.zeros(L, L)
        w[self.i_x, self.i_y] = self.w
        w[self.i_y, self.i_x] = self.w
        w = self.sig(w)

        L_inds = torch.arange(L).long()
        L_inds = L_inds[None, :]
        x1_inds = torch.arange(n1).long()
        x1_inds = x1_inds[:, None].expand(n1, L)
        x2_inds = torch.arange(n2).long()
        x2_inds = x2_inds[:, None].expand(n2, L)

        S1 = self.network(X1).view(n1, L, self.n_aa, -1)
        S1 = torch.matmul(S1, S1.transpose(-1, -2))
        S2 = self.network(X2).view(n2, L, self.n_aa, -1)
        S2 = torch.matmul(S2, S2.transpose(-1, -2))
        subs = S1[x1_inds, L_inds.expand(n1, L), X1, X1]
        k1 = self.wdk(subs, w).view((n1, 1))
        subs = S2[x2_inds, L_inds.expand(n2, L), X2, X2]
        k2 = self.wdk(subs, w).unsqueeze(0)

        inds1 = torch.arange(n1).long()[:, None]
        inds1 = inds1.expand(n1, n2).contiguous().view(-1)
        X1m = torch.index_select(X1, 0, inds1)
        inds2 = torch.arange(n2).long()
        inds2 = inds2.repeat(n1)
        X2m = torch.index_select(X2, 0, inds2)
        L_inds = L_inds.expand(n1 * n2, L)
        subs = S1[inds1[:, None].expand(n1 * n2, L), L_inds, X1m, X2m]
        K = self.wdk(subs, w).view((n1, n2))
        subs = S2[inds2[:, None].expand(n1 * n2, L), L_inds, X1m, X2m]
        K += self.wdk(subs, w).view((n1, n2))
        K /= 2
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K

class SeriesWDK(WeightedDecompositionKernel):
    """ No attention, L substitution matrices. """

    def __init__(self, contacts, L, n_S, d, a=1.0):
        super(WeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        A = torch.randn((L, n_S, d))
        self.A = Parameter(A)
        self.graph = self.make_graph(contacts, L)
        self.graph.to(self.A.device)
        self.n_S = n_S

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()
        S = torch.bmm(self.A, self.A.transpose(-1, -2))
        L_inds = torch.arange(L).long()
        subs = S[L_inds, X1, X1]
        k1 = self.wdk(subs).view((n1, 1))
        subs = S[L_inds, X2, X2]
        k2 = self.wdk(subs).unsqueeze(0)
        subs = S[L_inds, X1][:, L_inds, X2].view((n1 * n2, L))
        K = self.wdk(subs).view((n1, n2))
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K


class FixedWDK(WeightedDecompositionKernel):

    def __init__(self, contacts, L, S, a=1.0):
        super(WeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        self.S = S
        self.graph = self.make_graph(contacts, L)
        self.graph.to(self.S.device)
        self.dic = {}

    def forward(self, X1, X2):
        if (repr(X1), repr(X2)) in self.dic:
            K = self.dic[repr(X1), repr(X2)]
        else:
            n1, L = X1.size()
            n2, _ = X2.size()
            subs = self.S[X1, X1]
            k1 = self.wdk(subs).view((n1, 1))
            subs = self.S[X2, X2]
            k2 = self.wdk(subs).unsqueeze(0)
            L_inds = torch.arange(L).long()
            subs = self.S[X1][:, L_inds, X2].view((n1 * n2, L))
            K = self.wdk(subs).view((n1, n2))
            K = (K / torch.sqrt(k1) / torch.sqrt(k2))
            self.dic[repr(X1), repr(X2)] = K
        return (self.a ** 2) * K


class SoftSeriesWDK(SoftWeightedDecompositionKernel):
    """ Pairwise attention, L substitution matrices."""

    def __init__(self, L, n_S, sub_dim, a=1.0):
        super(SoftWeightedDecompositionKernel, self).__init__()
        self.a = Parameter(torch.tensor([a]))
        A = torch.randn((L, n_S, sub_dim))
        self.A = Parameter(A)
        self.n_S = n_S
        self.w = Parameter(torch.zeros(L * (L - 1) // 2))
        self.sig = nn.Sigmoid()
        i_x, i_y = np.tril_indices(L, k=-1)
        self.i_x = torch.tensor(i_x).long()
        self.i_y = torch.tensor(i_y).long()
        self.graph = torch.LongTensor([list(range(L)) for i in range(L)])
        self.graph = self.graph.to(self.A.device)

    def forward(self, X1, X2):
        n1, L = X1.size()
        n2, _ = X2.size()
        w = torch.zeros(L, L)
        w[self.i_x, self.i_y] = self.w
        w[self.i_y, self.i_x] = self.w
        w = self.sig(w)
        S = torch.bmm(self.A, self.A.transpose(-1, -2))
        L_inds = torch.arange(L).long()
        subs = S[L_inds, X1, X1]
        k1 = self.wdk(subs, w).view((n1, 1))
        subs = S[L_inds, X2, X2]
        k2 = self.wdk(subs, w).unsqueeze(0)
        subs = S[L_inds, X1][:, L_inds, X2].view((n1 * n2, L))
        K = self.wdk(subs, w).view((n1, n2))
        K = (K / torch.sqrt(k1) / torch.sqrt(k2))
        return (self.a ** 2) * K


class SumKernel(BaseKernel):

    def __init__(self, kernels):
        super(SumKernel, self).__init__()
        self.kernels = nn.ModuleList(kernels)

    def forward(self, X1, X2):
        K = torch.zeros(len(X1), len(X2))
        for k in self.kernels:
            K += k(X1, X2)
        return K
