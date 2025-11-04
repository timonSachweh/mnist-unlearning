import copy

from torch.nn.utils import vector_to_parameters

from train import run_training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import device

def retrain(model, train_loader, test_loader, num_epochs):
    run_training(model, train_loader, test_loader, epochs=num_epochs)


def unlearn(model, keep_data_loader: DataLoader, unlearn_loader: DataLoader, batch_size: int = 64,
            unlearn_epochs: int = 3, loss=nn.CrossEntropyLoss(), lambda_var: float = 0.1, learning_rate: float = 0.001):
    # compute gradients on unlearn_loader and adjust model weights accordingly
    removal_x, removal_y = _get_unlearn_data(unlearn_loader)
    model.to(device)

    grads = _calculate_gradients(model, removal_x, removal_y, loss)

    y_treshold = torch.median(grads.abs()).item()
    print("median absolute gradient (y_treshold):", y_treshold)

    saliency_map = (grads.abs() > y_treshold).float()
    print("Number of parameters to adjust:", int(saliency_map.sum().item()))

    theta_weights = torch.cat([p.data.view(-1) for p in model.parameters()])
    theta_g_weights = torch.cat([p.data.view(-1) for p in model.parameters()])

    model_working_copy = copy.deepcopy(model)
    model_working_copy.to(device)
    for epoch in range(unlearn_epochs):
        print(f"Unlearning epoch {epoch + 1}/{unlearn_epochs}")
        for i in range(int(max(len(keep_data_loader), len(unlearn_loader)) / batch_size)):
            keep_x, keep_y, unlearn_x, unlearn_y = _sample_batch_from_dataset(keep_data_loader, unlearn_loader,
                                                                              batch_size)
            unlearn_y_fake = _generate_fake_label(unlearn_y)
            keep_x = keep_x.to(device)
            keep_y = keep_y.to(device)
            unlearn_x = unlearn_x.to(device)
            unlearn_y_fake = unlearn_y_fake.to(device)

            masked_params = saliency_map * theta_weights

            vector_to_parameters(masked_params.detach().clone(), model_working_copy.parameters())
            model_working_copy.zero_grad()
            pred_unlearn = model_working_copy(unlearn_x)
            pred_learn = model_working_copy(keep_x)

            l = loss(pred_unlearn, unlearn_y_fake) + loss(pred_learn, keep_y) + lambda_var * (
                    masked_params - saliency_map * theta_g_weights).norm(p=2)
            l.backward()

            gradient = torch.cat(
                [p.grad.detach().view(-1) for p in model_working_copy.parameters() if p.grad is not None])
            theta_weights = theta_weights - gradient * learning_rate

    theta_weights = saliency_map * theta_weights + (1 - saliency_map) * theta_g_weights
    vector_to_parameters(theta_weights.detach().clone(), model.parameters())
    return model


# TODO: potentiell nicht alle zu vergessenden Punkte erwischt
def _sample_batch_from_dataset(keep_data_loader, unlearn_loader, batch_size):
    # sample batch_size random indices from the datasets (use dataset lengths, not loader lengths)
    keep_idx = torch.randint(0, len(keep_data_loader), (batch_size,)).tolist()
    unlearn_idx = torch.randint(0, len(unlearn_loader), (batch_size,)).tolist()

    # gather samples and stack into batch tensors
    keep_samples = [keep_data_loader.dataset[i] for i in keep_idx]
    keep_x = torch.stack([s[0] for s in keep_samples])
    keep_y = torch.tensor([s[1] for s in keep_samples])

    unlearn_samples = [unlearn_loader.dataset[i] for i in unlearn_idx]
    unlearn_x = torch.stack([s[0] for s in unlearn_samples])
    unlearn_y = torch.tensor([s[1] for s in unlearn_samples])
    return keep_x, keep_y, unlearn_x, unlearn_y


def _generate_fake_label(unlearn_y):
    # create fake labels in 0..10 that differ elementwise from unlearn_y
    unlearn_y_fake = torch.randint(0, 10, unlearn_y.shape, dtype=unlearn_y.dtype)
    eq_mask = (unlearn_y_fake == unlearn_y)
    if eq_mask.any():
        unlearn_y_fake[eq_mask] = (unlearn_y_fake[eq_mask] + 1) % 10
    return unlearn_y_fake


def _calculate_gradients(model, removal_x, removal_y, loss):
    model = model.to(device)
    removal_x = removal_x.to(device)
    removal_y = removal_y.to(device)

    model.zero_grad()
    out = model(removal_x)
    l = loss(out, removal_y)
    l.backward()

    g = [p.grad.detach().view(-1) for p in model.parameters() if p.grad is not None]

    if len(g) == 0:
        raise RuntimeError("No gradients were computed for the model parameters")
    return torch.cat(g)


def _get_unlearn_data(unlearn_loader: DataLoader):
    removal_x_batches = []
    removal_y_batches = []
    for xb, yb in unlearn_loader:
        removal_x_batches.append(xb)
        removal_y_batches.append(yb)
    if len(removal_x_batches) == 0:
        raise ValueError("unlearn_loader contains no data")

    removal_x = torch.cat(removal_x_batches, dim=0)
    removal_y = torch.cat(removal_y_batches, dim=0)
    return removal_x, removal_y
