import torch
import torch.nn as nn
import torch.nn.functional as f

class PSDPPGuide(nn.Module):
    """
        Guide with similarity between trajectories
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x_0):
        states = x_0[:, :, self.action_dim:self.action_dim+self.state_dim]
        phi = torch.flatten(states, start_dim=1)
        phi = f.normalize(phi, p=2, dim=1)
        S_B = torch.mm(phi, phi.t())
        det = torch.linalg.det(S_B)
        return torch.log(det)

    def gradients(self, x_prev, x_0_hat, *args):
        y = self(x_0_hat)
        grad = torch.autograd.grad([y.sum()], [x_prev])[0]
        y = y.expand(x_0_hat.shape[0])
        return y, grad
