import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from ml.evaluate import evaluate
from utils import device
from scipy.spatial import distance


def get_dataloaders(batch_size: int = 128, remove_label: int | None = None, remove_elements: int | None = None,
                    cifar: bool = False):
    if cifar:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if remove_elements is not None:
        # Randomly remove a certain number of elements from the training set
        if remove_elements > len(train_set):
            raise ValueError("remove_elements exceeds the size of the training set")
        indices = list(range(len(train_set)))
        np.random.shuffle(indices)
        indices_to_keep = indices[remove_elements:]
        incides_to_remove = indices[:remove_elements]
        new_train_set = Subset(train_set, indices_to_keep)
        removed_set = Subset(train_set, incides_to_remove)
        return DataLoader(new_train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size, shuffle=False), DataLoader(removed_set, batch_size=batch_size, shuffle=False), None

    if remove_label is not None:
        # Filter out all examples with the given label
        train_idx = [i for i, t in enumerate(train_set.targets) if int(t) != int(remove_label)]
        removed_train_data = [i for i in range(len(train_set)) if int(train_set[i][1]) == int(remove_label)]
        removed_test_data = [i for i in range(len(test_set)) if int(test_set[i][1]) == int(remove_label)]
        test_idx = [i for i, t in enumerate(test_set.targets) if int(t) != int(remove_label)]
        new_train_set = Subset(train_set, train_idx)
        new_test_set = Subset(test_set, test_idx)
        removed_train_set = Subset(train_set, removed_train_data)
        removed_test_set = Subset(test_set, removed_test_data)
        return DataLoader(new_train_set, batch_size=batch_size, shuffle=True), DataLoader(new_test_set, batch_size=batch_size, shuffle=False), DataLoader(removed_train_set, batch_size=batch_size, shuffle=False), DataLoader(removed_test_set, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, None, None


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        _, preds = out.max(1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def run_training(model, train_data, test_data, epochs: int = 3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_data, optimizer, criterion)
        val_loss, val_acc = evaluate(model, test_data, criterion)
        print(
            f"Epoch {epoch}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

    return model


def compute_softmax_output(model, loader):
    model.to(device)
    model.eval()
    softmax = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.log_softmax(out, dim=1)
            softmax.append(probs.cpu())
    return torch.cat(softmax)


def compute_distance(retrained_model, unlearned_model, loader):
    retrained_model.to(device)
    unlearned_model.to(device)
    softmax_retrained = compute_softmax_output(retrained_model, loader)
    softmax_unlearned = compute_softmax_output(unlearned_model, loader)
    distance_r_u = distance.jensenshannon(softmax_retrained, softmax_unlearned)
    return distance_r_u
