import argparse

from evaluate import evaluate_log
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
        model = run_training(model=model, train_data=train_data, test_data=test_data, epochs=args.epochs)
        evaluate_log(model, train_data, test_data, train_data_reduced, test_data_reduced, prefix="Initial training")
    

    if args.class_removed:
        model = run_training(model=model, train_data=train_data_reduced, test_data=test_data_reduced, epochs=args.epochs)
        evaluate_log(model, train_data, test_data, train_data_reduced, test_data_reduced, removed_train_data=removed_train_data, removed_test_data=removed_test_data, prefix="After removing class")

    if args.elements_removed:
        train_data_reduced, test_data, elements_removed, _ = get_dataloaders(batch_size=args.batch_size, remove_elements=args.elements)
        model = run_training(model=model, train_data=train_data_reduced, test_data=test_data, epochs=args.epochs)
        evaluate_log(model, train_data, test_data, train_data_reduced, elements_removed=elements_removed, prefix="After removing elements")



if __name__ == "__main__":
    main()
