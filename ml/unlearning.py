import copy

from torch.nn import MSELoss
from torch.nn.utils import vector_to_parameters

from .masked_forward import fast_functional_forward
from .train import run_training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import device


def retrain(model, train_loader, test_loader, num_epochs):
    run_training(model, train_loader, test_loader, epochs=num_epochs)


# paper uses NLLoss
def unlearn(model, keep_data_loader: DataLoader, unlearn_loader: DataLoader, batch_size: int = 64,
            unlearn_epochs: int = 3, loss=nn.NLLLoss(), lambda_var: float = 0.1, learning_rate: float = 0.001):
    # compute gradients on unlearn_loader and adjust model weights accordingly
    removal_x, removal_y = _get_unlearn_data(unlearn_loader)
    l2loss_norm = MSELoss(reduction="sum")
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
    theta_weights = torch.nn.Parameter(theta_weights.clone().detach().to(device))
    param_shapes = [p.shape for p in model_working_copy.parameters()]
    param_sizes = [p.numel() for p in model_working_copy.parameters()]

    for epoch in range(unlearn_epochs):
        #if (epoch + 1) % 10 == 0:
        #    print(f"Unlearning epoch {epoch + 1}/{unlearn_epochs}")
        print(f"Unlearning epoch {epoch + 1}/{unlearn_epochs}")
        for i in range(int(max(len(keep_data_loader), len(unlearn_loader)) / batch_size)):
            keep_x, keep_y, unlearn_x, unlearn_y = _sample_batch_from_dataset(keep_data_loader, unlearn_loader,
                                                                              batch_size)
            unlearn_y_fake = _generate_fake_label(unlearn_y)
            keep_x = keep_x.to(device)
            keep_y = keep_y.to(device)
            unlearn_x = unlearn_x.to(device)
            unlearn_y_fake = unlearn_y_fake.to(device)

            masked_params_flat = saliency_map * theta_weights
            split_tensors = torch.split(masked_params_flat, param_sizes)
            masked_params = [t.reshape(s) for t, s in zip(split_tensors, param_shapes)]

            # changed forward pass to use masked params
            pred_unlearn = fast_functional_forward(model, unlearn_x, masked_params)
            pred_learn = fast_functional_forward(model, keep_x, masked_params)

            # transform masked_params to a single tensor for norm calculation
            masked_params_tensor = torch.cat([p.view(-1) for p in masked_params])
            theta_g_tensor = (saliency_map * theta_g_weights).view(-1)

            if isinstance(loss, nn.NLLLoss):
                pred_unlearn = torch.log_softmax(pred_unlearn, dim=1)
                pred_learn = torch.log_softmax(pred_learn, dim=1)

            norm_loss = l2loss_norm(masked_params_tensor, saliency_map * theta_g_tensor)
            l = loss(pred_unlearn, unlearn_y_fake) + loss(pred_learn, keep_y) + lambda_var * norm_loss
            l.backward()

            with torch.no_grad():
                if theta_weights.grad is not None:
                    theta_weights -= learning_rate * theta_weights.grad
                    theta_weights.grad.zero_()
                else:
                    print("No gradients for theta_weights")

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
