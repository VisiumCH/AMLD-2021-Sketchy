import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """GRL Layer"""

    @staticmethod
    def forward(ctx, x, lambd=0.5):
        """lambd changes from 0 (only trains the classifier
        but does not update the encoder network) to 1
        """
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Reverse sign of gradient"""
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return ctx.lambd * grad_output.neg(), None


def grad_reverse(x, lambd=0.5):
    """ Reverse the sign of the gradient in the backward step """
    return GradReverse.apply(x, lambd)


class DomainLoss(nn.Module):
    """Ensures that embeddings belong to the same space"""

    def __init__(self, input_size=256, hidden_size=64):
        super(DomainLoss, self).__init__()
        self.input_size = input_size
        # self.map = nn.Linear(self.input_size, 1)
        self.map = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, target):
        x = self.map(x)
        x = torch.sigmoid(x).squeeze()
        return F.binary_cross_entropy(x, target)


class DetangledJoinDomainLoss(nn.Module):
    """
    Weighted Joined Triplet loss and Domain loss
    Triplet Loss: reduce the distance between embedded sketch and image
                  if they belong to the same class and
                  increase it if they belong to different classes.
    """

    def __init__(self, emb_size=256, w_dom=0.25, w_spa=0.25, lambd=0.5):
        super(DetangledJoinDomainLoss, self).__init__()

        self.emb_size = emb_size

        self.w_dom = w_dom
        self.w_spa = w_spa

        self.lambd = lambd

        self.domain_loss_mu = DomainLoss(input_size=int(self.emb_size))
        self.space_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, im_pos_sem, sk_sem, im_neg_sem, epoch):

        # Space Loss
        loss_spa = self.space_loss(sk_sem, im_pos_sem, im_neg_sem)

        # Domain Loss
        bz = sk_sem.size(0)
        targetSK = torch.zeros(bz)
        targetIM = torch.ones(bz)
        if sk_sem.is_cuda:
            targetSK = targetSK.cuda()
            targetIM = targetIM.cuda()

        # lmb = 0 (only train classifier) to lmb = 1 (only train encoder)
        if epoch > 25:
            lmb = 1.0
        elif epoch < 5:
            lmb = 0
        else:
            lmb = (epoch - 5) / 20.0
        loss_dom = self.domain_loss_mu(grad_reverse(sk_sem, lambd=lmb), targetSK)
        loss_dom += self.domain_loss_mu(grad_reverse(im_pos_sem, lambd=lmb), targetIM)
        loss_dom += self.domain_loss_mu(grad_reverse(im_neg_sem, lambd=lmb), targetIM)
        loss_dom = loss_dom / 3.0

        # Weighted Loss
        loss = self.w_dom * loss_dom + self.w_spa * loss_spa

        return loss, loss_dom, loss_spa
