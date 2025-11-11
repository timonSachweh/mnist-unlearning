import copy

import torch
import torch.nn as nn
from utils import device
import matplotlib.pyplot as plt
import numpy as np
from ml.unlearning import unlearn


def test_unlearning_over_lambdas(
        model,
        keep_loader,
        unlearn_loader,
        test_loader,
        batch_size=64,
        unlearn_epochs=12,
        loss=nn.CrossEntropyLoss(),
        learning_rate=0.001,
        lambda_steps=None,  # falls explizit gesetzt
        runs_per_lambda=3  # <--- Anzahl an Wiederholungen pro Lambda
):
    """
    Testet verschiedene Werte von λ.
    Für jeden Wert werden mehrere Unlearning-Läufe ausgeführt.
    Der Median der Accuracy wird geplottet.
    """

    # Liste der Lambdas aus dem main-Parser nutzen
    if isinstance(lambda_steps, list) or isinstance(lambda_steps, np.ndarray):
        lambdas = np.array(lambda_steps)
    else:
        raise ValueError("lambda_steps muss eine Liste der Lambdas sein.")

    run_accuracies = []

    for i, lam in enumerate(lambdas):
        print(f"\n=== Testing lambda = {lam:.3f} with {runs_per_lambda} runs ===")

        per_run_accuracies = []

        for run in range(runs_per_lambda):
            print(f"  -> run {run + 1}/{runs_per_lambda}")

            # Modellkopie erstellen
            model_copy = copy.deepcopy(model)

            # Unlearning
            model_copy = unlearn(
                model_copy,
                keep_loader,
                unlearn_loader,
                batch_size=batch_size,
                unlearn_epochs=unlearn_epochs,
                loss=loss,
                lambda_var=float(lam),
                learning_rate=learning_rate,
            )

            acc = evaluate_accuracy(model_copy, test_loader)
            print(acc)
            per_run_accuracies.append(acc)

        run_accuracies.append(per_run_accuracies)

        plot_lambda_scan(lambdas[:i + 1], run_accuracies, f"./images/lambda_results_temp{lam: .3f}.png")

    return lambdas, run_accuracies


def plot_lambda_scan(lambdas, run_accuracies, plot_path="./images/lambda_results.png"):
    plt.figure(figsize=(8, 5))
    # plt.ylim(0, 1)
    plt.boxplot(run_accuracies, positions=lambdas, widths=0.02)
    plt.xlabel("Lambda")
    plt.ylabel("Boxplot Test Accuracy")
    plt.title(f"Effect of λ on Unlearning Performance (Boxplot of {len(run_accuracies[0])} runs)")
    plt.grid(True)
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"\n✅ Plot gespeichert unter: {plot_path}")


def evaluate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total
