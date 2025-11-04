import argparse
import torch.nn as nn
from model import LeNet
from train import get_dataloaders, run_training, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",action="store_true",help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--class-removed",action="store_true",help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--elements-removed",action="store_true",help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--remove-label", type=int, default=1, help="Label to remove for second run")
    parser.add_argument("--elements", type=int, default=20, help="Number of elements to remove from training set")
    args = parser.parse_args()

    train_data, test_data, _, _ = get_dataloaders(batch_size=args.batch_size, remove_label=None)
    train_data_reduced, test_data_reduced, removed_train_data, removed_test_data = get_dataloaders(batch_size=args.batch_size, remove_label=args.remove_label)

    model = LeNet()

    if args.full:
        print("Training on full dataset...")
        model = run_training(model=model, train_data=train_data, test_data=test_data, epochs=args.epochs)
        val_loss, val_acc = evaluate(model, train_data, nn.CrossEntropyLoss())
        print(f"Final evaluation on train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_data, nn.CrossEntropyLoss())   
        print(f"Final evaluation on test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, train_data_reduced, nn.CrossEntropyLoss())
        print(f"Final evaluation on train dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_data_reduced, nn.CrossEntropyLoss())   
        print(f"Final evaluation on test dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, removed_train_data, nn.CrossEntropyLoss())
        print(f"Final evaluation on removed train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, removed_test_data, nn.CrossEntropyLoss())   
        print(f"Final evaluation on removed test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
    

    if args.class_removed:
        print(f"Training with label {args.remove_label} removed...")
        model = run_training(model=model, train_data=train_data_reduced, test_data=test_data_reduced, epochs=args.epochs)
        val_loss, val_acc = evaluate(model, train_data, nn.CrossEntropyLoss())
        print(f"Final evaluation on train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_data, nn.CrossEntropyLoss())   
        print(f"Final evaluation on test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, train_data_reduced, nn.CrossEntropyLoss())
        print(f"Final evaluation on train dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_data_reduced, nn.CrossEntropyLoss())   
        print(f"Final evaluation on test dataset reduced - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, removed_train_data, nn.CrossEntropyLoss())
        print(f"Final evaluation on removed train dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, removed_test_data, nn.CrossEntropyLoss())   
        print(f"Final evaluation on removed test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")


    if args.elements_removed:
        print(f"Training with data where {args.elements} elements are removed from dataset")
        train_data, test_data, elements_removed, _ = get_dataloaders(batch_size=args.batch_size, remove_elements=args.elements)
        model = run_training(model=model, train_data=train_data, test_data=test_data, epochs=args.epochs)
        val_loss, val_acc = evaluate(model, train_data, nn.CrossEntropyLoss())
        print(f"Final evaluation on train dataset without elements - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_data, nn.CrossEntropyLoss())   
        print(f"Final evaluation on test dataset - val loss {val_loss:.4f} acc {val_acc:.4f}")
        val_loss, val_acc = evaluate(model, elements_removed, nn.CrossEntropyLoss())
        print(f"Final evaluation on dataset elements removed - val loss {val_loss:.4f} acc {val_acc:.4f}")



if __name__ == "__main__":
    main()
