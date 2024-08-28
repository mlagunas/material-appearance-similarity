import torch
from torch import nn
from tqdm import tqdm

inf = float('inf')


def _diag_expand(tensor):
    return torch.diag(tensor).view(-1, 1).expand_as(tensor)


def _diag_to_matrix(diag):
    return diag.expand(len(diag), len(diag)) * torch.eye(len(diag)).to(
        diag.device)


def _eigen_to_matrix(l, v):
    """ v: eigenvectors, l:eigenvalues as returned by torch.eig"""

    # get index of significant eigenvectors/values
    ind = (l[..., 0] > 0).nonzero().squeeze()
    assert len(
        ind) > 0, 'Projection onto PSD cone failed. All eigenvalues were negative.'

    v = (v[:, ind])  # get only significant vectors
    l = _diag_to_matrix(l[ind, 0])
    k = v @ l @ v.t()

    assert not torch.any(torch.isinf(
        k)), 'Projection onto PSD cone failed. Metric contains Inf values.'
    assert not torch.any(torch.isnan(
        k)), 'Projection onto PSD cone failed. Metric contains NaN values.'

    return k


class cklXLoss(nn.Module):
    def __init__(self, mu, eps=1e-8):
        super(cklXLoss, self).__init__()
        self.mu = mu
        self.eps = eps

    def forward(self, x, triplets):
        sum_x = (x ** 2).sum(dim=1).expand(x.shape[0], x.shape[0])

        # get the gram matrix difference between pairs
        d_x = (-2 * (x @ x.t()) + sum_x + sum_x.t())

        # get probability
        numer = self.mu + d_x[triplets[:, 0], triplets[:, 2]]
        denom = 2 * self.mu + d_x[triplets[:, 0], triplets[:, 1]] + d_x[
            triplets[:, 0], triplets[:, 2]]

        # numerical stability
        numer = torch.clamp(numer, min=self.eps)
        denom = torch.clamp(denom, min=self.eps)

        # apply log rule and get loss
        loss = (torch.log(numer) - torch.log(denom)).sum()
        return -loss


class cklXSolver():
    def __init__(self, mu, lr=1e-1, max_iter=1000, tol=1e-8, device=None):
        self.mu = mu
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if device is not None else 'cpu'

    def fit(self, triplets, ndims=2, init_x=None):
        """
        :param triplets: array with triplets of indexes. Each index corresponds to an input.
        :param mu: paramter of the crowd kernel learning (ckl) algorithm
        :param ndims: number of output dimensions
        :param init_x: how to initialize X matrix

        :return:
        """

        ckl_loss = cklXLoss(mu=self.mu)

        n = len(torch.unique(triplets))  # maximum number of images
        n_triplets = len(triplets)  # number of triplets samples

        # initialized the k matrix
        if init_x is not None:
            x = init_x
        else:
            x = torch.rand((n, ndims)) * 0.01
            x.requires_grad_()

        # initialize some variables to keep track of the test
        best_x = inf
        best_loss = inf
        loss = inf
        with torch.set_grad_enabled(True):
            iter = 0;
            no_incr = 0;
            pbar = tqdm(total=self.max_iter)
            while iter < self.max_iter and no_incr < 5:
                old_c = loss

                # perform gradient update
                loss = ckl_loss(x, triplets)
                loss.backward()
                with torch.no_grad():
                    x -= (self.lr / n_triplets * n) * x.grad
                    # Manually zero the gradients after running the backward pass
                    x.grad.zero_()

                    # keep track of the best solution
                    if loss < best_loss:
                        best_loss = loss
                        best_x = x

                    # update training parameters
                    if old_c > loss + self.tol:
                        no_incr = 0
                        self.lr *= 1.01
                    else:
                        no_incr += 1
                        self.lr *= .5

                    # print progress
                    sum_x = (x ** 2).sum(dim=1).expand(x.shape[0], x.shape[0])

                    # get the difference gram matrix between pairs
                    d_k = (-2 * (x @ x.t()) + sum_x + sum_x.t())
                    n_viol = (d_k[triplets[:, 0], triplets[:, 1]] > d_k[
                        triplets[:, 0], triplets[:, 2]]).sum().item()
                    pbar.set_description(
                        '[mu: %2.2f] error is %.4f | number of constraints %d (%.4f %s) '
                        % (self.mu, loss, n_viol, n_viol / n_triplets, '%'))
                    pbar.update()
                    # print('[iter] %d | error is %.4f | number of constraints %d (%.4f %s) '
                    #       % (iter, loss, n_viol, n_viol / n_triplets, '%'))
                    iter += 1

        # return best values
        x = best_x
        return x


class cklKLoss(nn.Module):
    def __init__(self, mu, eps=1e-8):
        super(cklKLoss, self).__init__()
        self.mu = mu
        self.eps = eps

    def forward(self, k, triplets):
        loss = self.log_prob(k, triplets).sum()
        return loss

    def log_prob(self, k, triplets):
        diag_k = _diag_expand(k)

        # get the gram matrix difference between pairs
        d_k = (-2 * k + diag_k + diag_k.t())

        # get probability
        numer = self.mu + d_k[triplets[:, 0], triplets[:, 2]]
        denom = 2 * self.mu + d_k[triplets[:, 0], triplets[:, 1]] + d_k[
            triplets[:, 0], triplets[:, 2]]

        # numerical stability
        numer = torch.clamp(numer, min=self.eps)
        denom = torch.clamp(denom, min=self.eps)

        # apply log rule and get loss
        likelihood = -((torch.log(numer) - torch.log(denom)))
        return likelihood


class cklKSolver():
    """
    optimizes the K matrix using the method proposed in "Adaptively Learning the Crowd Kernel"
    """

    def __init__(self, mu, lr=1e-1, max_iter=1000, tol=1e-8, device=None):
        self.mu = mu
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.device = device if device is not None else 'cpu'

    def fit(self, triplets, ndims=2, init_k=None):
        """
        :param triplets: array with triplets of indexes. Each index corresponds to an input.
        :param mu: paramter of the crowd kernel learning (ckl) algorithm
        :param ndims: number of output dimensions
        :param init_k: how to initialize K matrix
        :return:
        """

        ckl_loss = cklKLoss(mu=self.mu)

        self.n = len(torch.unique(triplets))  # maximum number of images
        n_triplets = len(triplets)  # number of triplets samples

        # initialized the k matrix
        if init_k is not None:
            k = init_k
        else:
            k = torch.rand((self.n, ndims)).to(self.device)
            k = (k @ k.t()).requires_grad_()

        # initialize some variables to keep track of the test
        best_k = inf
        best_loss = inf
        loss = inf

        with torch.set_grad_enabled(True):
            iter = 0;
            no_incr = 0;
            pbar = tqdm(total=self.max_iter)

            while iter < self.max_iter and no_incr < 10:
                old_c = loss

                # perform gradient update
                loss = ckl_loss(k, triplets)
                loss.backward()
                with torch.no_grad():
                    k -= (self.lr / n_triplets * self.n) * k.grad

                    # Manually zero the gradients after running the backward pass
                    k.grad.zero_()

                    # project back kernel onto PSD cone
                    l, v = torch.eig(k, eigenvectors=True)
                k = _eigen_to_matrix(l, v).requires_grad_()

                # keep track of the best solution
                if loss < best_loss:
                    best_loss = loss
                    best_k = k

                # update training parameters
                if old_c > loss + self.tol:
                    no_incr = 0
                    self.lr *= 1.01
                else:
                    no_incr += 1
                    self.lr *= .5

                # print progress
                if iter % 10 == 0:
                    diag_k = _diag_expand(k)

                    # get the difference gram matrix between pairs
                    d_k = (-2 * k + diag_k + diag_k.t())
                    n_viol = (d_k[triplets[:, 0], triplets[:, 1]] > d_k[
                        triplets[:, 0], triplets[:, 2]]).sum().item()

                    pbar.set_description(
                        '[mu: %2.2f] error is %.4f | number of constraints %d (%.4f %s) '
                        % (self.mu, loss, n_viol, n_viol / n_triplets, '%'))
                    pbar.update()
                iter += 1

        # return best values
        k = best_k

        # recover x with the given ndims
        x, l, _ = torch.svd(k)
        x = x[:, :ndims]
        l = (l[:ndims]).expand_as(x)
        x = torch.sqrt(l) * x

        return x
