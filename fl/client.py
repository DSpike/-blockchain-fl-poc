"""Federated learning client: local training."""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    def __init__(self, client_id, model, train_loader: DataLoader,
                 lr=0.01, local_epochs=5, device="cpu"):
        self.client_id    = client_id
        self.model        = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.lr           = lr
        self.local_epochs = local_epochs
        self.device       = device

    def set_global_weights(self, global_state_dict):
        self.model.load_state_dict(copy.deepcopy(global_state_dict))

    def train(self):
        self.model.train()
        optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr,
                                    momentum=0.9, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        for _ in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimiser.step()
        return self.model
