import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.optim as optim

from torch.utils.checkpoint import checkpoint


class NLMLLoss(nn.Module):
    """ The unnormalized negative log marginal likelihood loss. """

    def __init__(self):
        super(NLMLLoss, self).__init__()

    def forward(self, L, alpha, y):
        first = 0.5 * y.t() @ alpha
        second = torch.sum(torch.log(torch.diag(L)))
        return first + second


class GPRegressor(nn.Module):

    def __init__(self, kernel, sn=0.1, lr=1e-1, scheduler=None, prior=True):
        super(GPRegressor, self).__init__()
        self.sn = Parameter(torch.Tensor([sn]))
        self.kernel = kernel
        self.loss_func = NLMLLoss()
        opt = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(opt, lr=lr)
        if prior:
            self.prior = torch.distributions.Beta(2, 2).log_prob
        else:
            self.prior = None
        if scheduler is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, *scheduler)
        else:
            self.scheduler = None

    def loss(self, X, y, jitter):
        K = self.kernel(X, X)
        inds = list(range(len(K)))
        K[[inds], [inds]] += self.sn + jitter
        L = torch.potrf(K, upper=False)
        alpha = torch.trtrs(y, L, upper=False)[0]
        alpha = torch.trtrs(alpha, L.t(), upper=True)[0]
        loss = self.loss_func(L, alpha, y)
        if self.prior is not None:
            loss -= self.prior(self.sn)
        return loss

    def forward(self, X):
        """ Gaussian process regression predictions.

        Parameters:
            X: m x d points to predict

        Returns:
            mu: m x 1 predicted means
            var: m x m predicted covariance

        Follows Algorithm 2.1 from GPML.
        """
        ### Implement prior ###
        ### Scaling
        k_star = self.kernel(self.X, X)
        mu =  k_star.t() @ self.alpha
        v = torch.trtrs(k_star, self.L, upper=False)[0]
        k_ss = self.kernel(X, X)
        var = k_ss - v.t() @ v
        return mu, var

    def fit(self, X, y, its=100, jitter=1e-6, verbose=True):
        self.X = X
        self.y = y
        self._fit(X, y, its, jitter, verbose)
        self._set_pars(jitter)
        return self.history

    def _fit(self, X, y, its, jitter, verbose):
        self.history = []
        for it in range(its):
            loss = self.loss(X, y, jitter)
            # backward
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            # update parameters
            self.optimizer.step()
            self.sn.data.clamp_(min=1e-6)
            if self.scheduler is not None:
                self.scheduler.step()
            if verbose:
                update = '\rIteration %d of %d\tNLML: %.4f\tsn: %.6f\t' \
                        %(it + 1, its, loss, self.sn.cpu().detach().numpy()[0])
                print(update, end='')
            self.history.append(loss.cpu().detach().numpy()[0][0])

    def _set_pars(self, jitter):
        Ky = self.kernel(self.X, self.X)
        inds = list(range(len(Ky)))
        Ky[[inds], [inds]] += self.sn + jitter
        self.L = torch.potrf(Ky, upper=False)
        self.alpha = torch.trtrs(self.y, self.L, upper=False)[0]
        self.alpha = torch.trtrs(self.alpha, self.L.t(), upper=True)[0]


class DeepGPRegressor(GPRegressor):

    def __init__(self, network, kernel, sn=0.1, lr=1e-1, scheduler=None, prior=True):
        super(GPRegressor, self).__init__()
        self.sn = Parameter(torch.Tensor([sn]))
        self.network = network
        self.kernel = kernel
        self.loss_func = NLMLLoss()
        opt = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(opt, lr=lr)
        if prior:
            self.prior = torch.distributions.Beta(2, 2).log_prob
        else:
            self.prior = None
        if scheduler is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, *scheduler)
        else:
            self.scheduler = None


    def forward(self, X):
        embedded = self.network(X)
        return super(DeepGPRegressor, self).forward(embedded)

    def loss(self, X, y, jitter):
        emb = self.network(X)
        return super(DeepGPRegressor, self).loss(emb, y, jitter)


    def fit(self, X, y, its=100, jitter=1e-6, verbose=True):
        self._fit(X, y, its, jitter, verbose)
        self.X = self.network(X)
        self.y = y
        self._set_pars(jitter)
        return self.history
