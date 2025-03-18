import torch
from torch import nn


class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.embed_dim = 256
        self.neck_dim = 8
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.neck_dim),
            nn.BatchNorm1d(8,momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.neck_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim,momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1,self.input_dim))
        return self.decoder(z), z
