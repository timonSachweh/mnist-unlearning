import argparse
import torch.nn as nn
from train import get_dataloaders, run_training, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--remove-label", type=int, default=1, help="Label to remove for second run")
    args = parser.parse_args()

    print("Training on full dataset...")
    train_data, test_data = get_dataloaders(batch_size=args.batch_size, remove_label=None)
    train_data_reduced, test_data_reduced = get_dataloaders(batch_size=args.batch_size, remove_label=args.remove_label)


    model = run_training(train_data=train_data, test_data=test_data, epochs=args.epochs)
    val_loss, val_acc = evaluate(model, train_data, nn.CrossEntropyLoss())
    print(f"Final evaluation on train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, test_data, nn.CrossEntropyLoss())   
    print(f"Final evaluation on test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, train_data_reduced, nn.CrossEntropyLoss())
    print(f"Final evaluation on train dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, test_data_reduced, nn.CrossEntropyLoss())   
    print(f"Final evaluation on test dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
    


    print(f"Training with label {args.remove_label} removed...")
    model = run_training(train_data=train_data_reduced, test_data=test_data_reduced, epochs=args.epochs)
    val_loss, val_acc = evaluate(model, train_data, nn.CrossEntropyLoss())
    print(f"Final evaluation on train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, test_data, nn.CrossEntropyLoss())   
    print(f"Final evaluation on test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, train_data_reduced, nn.CrossEntropyLoss())
    print(f"Final evaluation on train dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
    val_loss, val_acc = evaluate(model, test_data_reduced, nn.CrossEntropyLoss())   
    print(f"Final evaluation on test dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")


if __name__ == "__main__":
    main()
