import copy

import torch
import torch.nn as nn

from ml.evaluate import evaluate
from ml.train import compute_distance, compute_zrf_score, ain
from utils import device
import matplotlib.pyplot as plt
import numpy as np
from ml.unlearning import unlearn


def test_unlearning_over_lambdas(
        model,
        keep_loader,
        unlearn_loader,
        test_loader,
        train_loader,
        batch_size=64,
        unlearn_epochs=12,
        loss=nn.NLLLoss(),
        learning_rate=0.001,
        lambda_steps=None,  # falls explizit gesetzt
        runs_per_lambda=3,  # <--- Anzahl an Wiederholungen pro Lambda
        distance=False,
        retrained_model=None,
        ain_b=False,
):
    """
    Testet verschiedene Werte von λ.
    Für jeden Wert werden mehrere Unlearning-Läufe ausgeführt.
    Der Median der Accuracy wird geplottet.
    """
    model_init = copy.deepcopy(model)
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

            if retrained_model is not None and distance:
                dist = compute_distance(retrained_model, model_copy, test_loader)
                zrf = compute_zrf_score(dist)
                plot_distance(dist, lam)
                print(f"Distance between retrained and unlearned model: {dist} and zrf score: {zrf}")

            if retrained_model is not None and ain_b:
                _, full_acc_forget = evaluate(model_init, unlearn_loader, loss)
                print("Accuracy of fully trained model on forget set is: {}".format(full_acc_forget))
                ain(model_init, model_copy, retrained_model, train_loader, unlearn_loader, keep_loader, loss=loss)

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


def plot_distance(distances, l):
    plt.figure(figsize=(8, 4))
    #plt.plot(distances, marker='o')  # Linienplot mit Punkten
    plt.scatter(range(len(distances)), distances)

    plt.xlabel("Classification class")
    plt.ylabel("JS-Distance")
    plt.title("JS-Distance between unlearned and retrained model")
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 0.2)
    plt.savefig(f"./images/js_distance_lambda_{l:.3f}.png", dpi=200)
    plt.close()


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
