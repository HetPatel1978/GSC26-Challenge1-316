"""Flower NumPyClient + shared train/test/(de)serialization helpers."""
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train(model: nn.Module, loader: DataLoader, epochs: int, lr: float, device: torch.device) -> None:
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for _ in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def test(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        local_epochs: int,
        lr: float,
        is_malicious: bool,
    ):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr
        self.is_malicious = is_malicious

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        return get_parameters(self.model)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        train(self.model, self.trainloader, self.local_epochs, self.lr, self.device)
        return get_parameters(self.model), len(self.trainloader.dataset), {"is_malicious": self.is_malicious}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_parameters(self.model, parameters)
        loss, acc = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}
