import torch
import torch.nn.functional as F


def forward_with_masked_params_lenet(x, masked_params):
    """
    Führt einen Forward-Pass für LeNet durch,
    wobei masked_params eine Liste der Parameter-Tensoren (in richtiger Reihenfolge) ist.
    """
    # Die Reihenfolge entspricht model.parameters() bei LeNet
    (
        conv1_weight, conv1_bias,
        conv2_weight, conv2_bias,
        fc1_weight, fc1_bias,
        fc2_weight, fc2_bias,
        fc3_weight, fc3_bias
    ) = masked_params

    # --- Feature extractor ---
    x = F.conv2d(x, conv1_weight, conv1_bias)
    x = torch.tanh(x)
    x = F.avg_pool2d(x, 2)
    x = F.conv2d(x, conv2_weight, conv2_bias)
    x = torch.tanh(x)
    x = F.avg_pool2d(x, 2)

    # --- Classifier ---
    x = torch.flatten(x, 1)
    x = F.linear(x, fc1_weight, fc1_bias)
    x = torch.tanh(x)
    x = F.linear(x, fc2_weight, fc2_bias)
    x = torch.tanh(x)
    x = F.linear(x, fc3_weight, fc3_bias)
    return x
