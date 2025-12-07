import copy
from itertools import cycle

import pandas as pd
import torch
import torch.nn as nn

from ml.train import compute_distance
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
        runs_per_lambda=3,  # <--- Anzahl an Wiederholungen pro Lambda
        distance=False,
        retrained_model=None,
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
    distance_results = {}

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
                # plot_distance(dist, lam)
                print(f"Distance between retrained and unlearned model: {dist}")
                if lam not in distance_results:
                    distance_results[lam] = []
                distance_results[lam].append(dist)

            acc = evaluate_accuracy(model_copy, test_loader)
            print(acc)
            per_run_accuracies.append(acc)
            if distance:
                plot_distance_runs(distance_results,
                                   save_path=f"./images/distances_lambda_{lam:.3f}.png",
                                   title="Jensen-Shannon Distance over λ")

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
    plt.title(f"JS-Distance between unlearned and retrained model with lambda = {l:.3f}")
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


def plot_distance_runs(distances_dict, save_path=None, title="Distances per λ"):
    """
    distances_dict erwartet:
         {lambda: [d1, d2, ...]}
    Diese Version ist fehlertolerant und wandelt Singles in Listen um.
    """

    # Farben definieren
    colors = cycle(plt.cm.tab10.colors)

    plt.figure(figsize=(10, 6))

    table_rows = []
    table_index = []

    for lam, vals in distances_dict.items():

        # --- WICHTIG: Sicherstellen, dass es eine Liste ist ---
        if np.isscalar(vals):
            vals = [vals]
        elif not isinstance(vals, (list, tuple, np.ndarray)):
            raise ValueError(f"Ungültiger Eintrag für λ={lam}: {vals}")

        dist_list = list(vals)

        # x-Werte müssen gleiche Länge haben wie dist_list
        x = np.full(len(dist_list), lam)

        plt.scatter(x, dist_list, s=60, color=next(colors), label=f"λ={lam}")

        table_rows.append(dist_list)
        table_index.append(f"λ={lam}")

    # Achsen & Titel
    plt.xlabel("λ")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Tabelle unter dem Plot
    df = pd.DataFrame(table_rows, index=table_index)
    plt.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=[f"Run {i+1}" for i in range(df.shape[1])],
        loc="bottom",
        cellLoc="center"
    )

    plt.subplots_adjust(bottom=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()