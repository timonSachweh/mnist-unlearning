

import torch
import torch.nn as nn
from utils import device


def evaluate(model, loader, criterion):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total


def evaluate_log(model, train_data, test_data, train_data_reduced, test_data_reduced=None, elements_removed=None, removed_train_data=None, removed_test_data=None, prefix="", loss=nn.CrossEntropyLoss()):
    print(f"----------------------- Results for {prefix} -------------------------------------------")
    val_loss, val_acc = evaluate(model, train_data, loss)
    print(f"train dataset: \t\t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, test_data, loss)   
    print(f"test dataset: \t\t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, train_data_reduced, loss)
    print(f"train dataset reduced: \t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    if test_data_reduced is not None:
        val_loss, val_acc = evaluate(model, test_data_reduced, loss)   
        print(f"test dataset reduced: \t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    if removed_train_data is not None:
        val_loss, val_acc = evaluate(model, removed_train_data, loss)
        print(f"removed train dataset: \t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    if removed_test_data is not None:
        val_loss, val_acc = evaluate(model, removed_test_data, loss)
        print(f"removed test dataset: \t\tloss {val_loss:.4f} acc {val_acc:.4f}")
    if elements_removed is not None:
        val_loss, val_acc = evaluate(model, elements_removed, loss)
        print(f"elements removed dataset: \tloss {val_loss:.4f} acc {val_acc:.4f}")
    print("-------------------------------------------------------------------------------------")