import copy
import os
from datetime import datetime
from itertools import cycle

import pandas as pd
import torch
import torch.nn as nn

from ml.evaluate import evaluate
from ml.train import compute_distance, ain, compute_zrf_score
from utils.profiling import timing_decorator
from utils_file import device
import matplotlib.pyplot as plt
import numpy as np
from ml.unlearning import unlearn


@timing_decorator("Testing unlearning over lambdas")
def test_unlearning_over_lambdas(
        model,
        keep_loader,
        unlearn_loader,
        test_loader,
        train_loader,
        batch_size=64,
        unlearn_epochs=12,
        loss=nn.CrossEntropyLoss(),
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
    _, full_acc_forget = evaluate(model_init, unlearn_loader, loss)
    # Liste der Lambdas aus dem main-Parser nutzen
    if isinstance(lambda_steps, list) or isinstance(lambda_steps, np.ndarray):
        lambdas = np.array(lambda_steps)
    else:
        raise ValueError("lambda_steps muss eine Liste der Lambdas sein.")

    run_accuracies = []
    distance_results = {}
    distance_orig_results = {}

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

            if retrained_model is not None:
                plot_retrain_unlearn_times(
                    csv_path="timing_results.csv",
                    plot_path="./images/retrain_unlearn_box.png",
                    show_boxplot=True
                )
                plot_retrain_unlearn_times(
                    csv_path="timing_results.csv",
                    plot_path="./images/retrain_unlearn_mean.png",
                    show_boxplot=False
                )

            if retrained_model is not None and distance:
                _, full_acc_forget = evaluate(model_init, unlearn_loader, loss)
                dist = compute_distance(retrained_model, model_copy, test_loader)
                zrf = compute_zrf_score(dist)
                dist_orig = compute_distance(model_init, model_copy, test_loader)
                zrf_orig = compute_zrf_score(dist_orig)
                # plot_distance(dist, lam)
                print(f"Distance between retrained and unlearned model: {dist} and zrf score: {zrf}")
                if lam not in distance_results:
                    distance_results[lam] = []
                    distance_orig_results[lam] = []
                distance_results[lam].append(dist)
                distance_orig_results[lam].append(dist_orig)

            if retrained_model is not None and ain_b:
                model_ain = copy.deepcopy(model)
                _, full_acc_forget = evaluate(model_ain, unlearn_loader, loss)
                print("Accuracy of fully trained model on forget set is: {}".format(full_acc_forget))
                ain(model_ain, model_copy, retrained_model, train_loader, unlearn_loader, keep_loader, loss=loss)

            acc = evaluate_accuracy(model_copy, test_loader)
            print(acc)
            per_run_accuracies.append(acc)

            metrics = {
                "accuracy": acc,
                "distance_retrained": dist,        # darf LISTE sein!
                "distance_original": dist_orig,
                "zrf_retrained": zrf,
                "zrf_original": zrf_orig
            }

            append_metrics_long_format(
                csv_path="lambda_experiments_long.csv",
                lambda_val=lam,
                run_id=run,
                metrics=metrics,
                extra_info={
                    "unlearn_epochs": unlearn_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                }
            )

        run_accuracies.append(per_run_accuracies)

        plot_lambda_scan(lambdas[:i + 1], run_accuracies, f"./images/lambda_results_temp{lam: .3f}.png")

    plot_lambda_development(distance_results, mode="mean", title="Jenson-Shannon Distance between unlearned and "
                                                                 "retrained model per lambda",
                            save_path=f"./images/lambda_results_all_mean{lam: .3f}.png")
    plot_lambda_development(distance_orig_results, mode="mean",
                            title="Jenson-Shannon Distance between unlearned and original model per lambda",
                            save_path=f"./images/lambda_results_orig_all_mean{lam: .3f}.png")
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
    # plt.plot(distances, marker='o')  # Linienplot mit Punkten
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
        colLabels=[f"Run {i + 1}" for i in range(df.shape[1])],
        loc="bottom",
        cellLoc="center"
    )

    plt.subplots_adjust(bottom=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def plot_lambda_development(data, mode="mean", ylabel="Distance", title="Jenson-Shannon Distance per lambda",
                            save_path=None):
    """
        Plottet die Entwicklung der Ergebnisse über verschiedene Lambdas hinweg.

        Parameters
        ----------
        data : dict
            Struktur:
            {
                lambda1 : [list_of_values_run1, list_of_values_run2, ...],
                lambda2 : [...],
                ...
            }

            Beispiel:
            {
                0.1: [[1,2,3], [2,3,4]],
                0.2: [[2,2,4], [3,4,5]]
            }

        mode : str
            "mean"       → Mittelwert + Standardabweichung
            "boxplot"    → Boxplot je Lambda
            "raw"        → alle Einzelkurven zeigen

        ylabel : str
            Achsenbeschriftung (y)

        title : str
            Plot-Titel
            :param data:
            :param mode:
            :param ylabel:
            :param title:
            :param save_path:
        """

    lambdas = sorted(data.keys())
    plt.figure(figsize=(10, 6))

    if mode == "mean":
        means = []
        stds = []
        for lam in lambdas:
            values = np.array([np.mean(run) for run in data[lam]])
            means.append(np.mean(values))
            stds.append(np.std(values))

        plt.errorbar(lambdas, means, yerr=stds, fmt='-o', capsize=5)
        plt.ylabel(ylabel)
        plt.xlabel("Lambda")
        plt.title(title)

    elif mode == "raw":
        for lam in lambdas:
            for run in data[lam]:
                plt.plot([lam] * len(run), run, 'o', alpha=0.3)

        plt.ylabel(ylabel)
        plt.xlabel("Lambda")
        plt.title(title)

    elif mode == "boxplot":
        all_values = [np.concatenate(data[lam]) for lam in lambdas]
        plt.boxplot(all_values, labels=lambdas)
        plt.ylabel(ylabel)
        plt.xlabel("Lambda")
        plt.title(title)

    else:
        raise ValueError("Unbekannter mode. Nutze: 'mean', 'boxplot', 'raw'.")

    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"\n✅ Plot gespeichert unter: {save_path}")
    # plt.show()


def plot_retrain_unlearn_times(csv_path, plot_path="./images/retrain_unlearn_times.png",
                               show_boxplot=True, target_time=None):
    df = pd.read_csv(csv_path)

    # Zeiten extrahieren
    retrain_times = df[df['function_name'].str.lower() == 'retrain']['duration_s'].tolist()
    unlearn_times = df[df['function_name'].str.lower() == 'unlearn']['duration_s'].tolist()

    if show_boxplot:
        # Boxplot
        plt.figure(figsize=(6, 5))
        plt.boxplot([retrain_times, unlearn_times], labels=["Retraining", "Unlearning"], patch_artist=True)
        plt.ylabel("Time (s)")
        plt.title(f"Comparison of retraining vs unlearning times in {len(retrain_times)} runs")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    else:
        # Balkenplot mit Mittelwert + Std
        means = [np.mean(retrain_times), np.mean(unlearn_times)]
        stds = [np.std(retrain_times), np.std(unlearn_times)]
        categories = ["Retraining", "Unlearning"]

        plt.figure(figsize=(6, 5))
        plt.bar(categories, means, yerr=stds, capsize=5, color=["tab:blue", "tab:orange"])
        plt.ylabel("Time (s)")
        plt.title("Mean time: Retraining vs Unlearning")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Optionale Ziel-Linie
    if target_time is not None:
        plt.axhline(y=target_time, color="red", linestyle="--", linewidth=2, label="Ziel-Zeit")
        plt.legend()

    # Speichern
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\n✅ Plot gespeichert unter: {plot_path}")


def append_metrics_long_format(
        csv_path,
        lambda_val,
        run_id,
        metrics: dict,
        extra_info: dict | None = None
):
    """
    Schreibt Metriken im LONG FORMAT in eine CSV-Datei.

    Jede Zeile entspricht genau einem numerischen Wert.

    Parameters
    ----------
    csv_path : str
        Ziel-CSV
    lambda_val : float
        Lambda-Wert
    run_id : int
        Run-ID
    metrics : dict
        {"accuracy": float,
         "distance_retrained": float | list | np.ndarray,
         ...}
    extra_info : dict, optional
        Zusätzliche Metadaten (epochs, lr, seed, ...)
    """

    rows = []
    timestamp = datetime.now().isoformat()

    for metric_name, metric_value in metrics.items():

        # Falls Liste / Array → EINZELNE Werte aufsplitten
        if isinstance(metric_value, (list, tuple, np.ndarray)):
            values = metric_value
        else:
            values = [metric_value]

        for v in values:
            if v is None:
                continue

            row = {
                "timestamp": timestamp,
                "lambda": float(lambda_val),
                "run": int(run_id),
                "metric": metric_name,
                "value": float(v),
            }

            if extra_info:
                row.update(extra_info)

            rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)

def plot_distance_mean_from_csv(
        csv_path,
        metric="distance_retrained",
        save_path="./images/distance_mean.png",
        title="Jensen-Shannon Distance per λ"
):
    df = load_experiment_csv(csv_path)

    # Nur gewünschte Metrik auswählen
    df = df[df["metric"] == metric]

    # Sicherstellen, dass value numerisch ist
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    grouped = df.groupby("lambda")["value"]

    means = grouped.mean()
    stds = grouped.std()

    plt.figure(figsize=(8, 5))
    plt.errorbar(means.index, means.values, yerr=stds.values, fmt='-o', capsize=5)
    plt.xlabel("Lambda")
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True)

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"✅ Distance-Plot gespeichert: {save_path}")


def plot_accuracy_mean_from_csv(
        csv_path,
        save_path="./images/accuracy_mean.png"
):
    df = load_experiment_csv(csv_path)

    # Accuracy filtern
    df = df[df["metric"] == "accuracy"]

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    grouped = df.groupby("lambda")["value"]

    means = grouped.mean()
    stds = grouped.std()

    plt.figure(figsize=(8, 5))
    plt.errorbar(means.index, means.values, yerr=stds.values, fmt='-o', capsize=5)
    plt.xlabel("Lambda")
    plt.ylabel("Test Accuracy")
    plt.title("Mean Test Accuracy ± Std per λ")
    plt.grid(True)

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"✅ Accuracy-Mean-Plot gespeichert: {save_path}")

def plot_accuracy_boxplot_from_csv(
            csv_path,
            save_path="./images/accuracy_boxplot.png"
    ):
    df = load_experiment_csv(csv_path)

    df = df[df["metric"] == "accuracy"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    lambdas = sorted(df["lambda"].unique())
    data = [
        df[df["lambda"] == lam]["value"].values
        for lam in lambdas
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, positions=lambdas, widths=0.02)
    plt.xlabel("Lambda")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy per λ (Boxplot)")
    plt.grid(True)

    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"✅ Accuracy-Boxplot gespeichert: {save_path}")

def load_experiment_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")
    return pd.read_csv(csv_path)


def create_all_plots_from_csv(csv_path):
    plot_accuracy_boxplot_from_csv(csv_path)
    plot_accuracy_mean_from_csv(csv_path)
    #plot_accuracy_scatter_from_csv(csv_path)

    plot_distance_mean_from_csv(
        csv_path,
        metric="distance_retrained",
        save_path="./images/distance_retrained_mean.png",
        title="Distance to retrained model per λ"
    )

    plot_distance_mean_from_csv(
        csv_path,
        metric="distance_original",
        save_path="./images/distance_original_mean.png",
        title="Distance to original model per λ"
    )


