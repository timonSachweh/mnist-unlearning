import argparse
import itertools
from typing import Iterable, Iterator, Union, Tuple, Dict

from evaluate import evaluate_log
from model import LeNet
from train import get_dataloaders, run_training, evaluate
from unlearning import unlearn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--class-removed", action="store_true",
                        help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--elements-removed", action="store_true",
                        help="If set, run the full-dataset training/evaluation section")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument('--unlearn-epochs', help='delimited list of unlearn-epochs', type=str, default='6')
    parser.add_argument('--unlearn-learning-rates', help='delimited list of unlearn-learning-rates', type=str,
                        default='0.01')
    parser.add_argument('--unlearn-batch-sizes', help='delimited list of unlearn-batch-sizes', type=str, default='64')
    parser.add_argument('--unlearn-lambdas', help='delimited list of unlearn-lambdas', type=str, default='0.01')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--remove-label", type=int, default=1, help="Label to remove for second run")
    parser.add_argument("--elements", type=int, default=20, help="Number of elements to remove from training set")
    args = parser.parse_args()

    ul_epochs = [int(i) for i in args.unlearn_epochs.split(',')]
    ul_learning_rates = [float(i) for i in args.unlearn_learning_rates.split(',')]
    ul_batch_sizes = [int(i) for i in args.unlearn_batch_sizes.split(',')]
    ul_lambdas = [float(i) for i in args.unlearn_lambdas.split(',')]

    d_train, d_test, _, _ = get_dataloaders(batch_size=args.batch_size, remove_label=None)
    d_cr_train, d_cr_test, d_cr_r_train, d_cr_r_test = get_dataloaders(batch_size=args.batch_size,
                                                                       remove_label=args.remove_label)

    model_init = run_training(model=LeNet(), train_data=d_train, test_data=d_test, epochs=args.epochs)
    evaluate_log(model_init, d_train, d_test, d_cr_train, d_cr_test, removed_train_data=d_cr_r_train,
                 removed_test_data=d_cr_r_test, prefix="Initial training")

    if args.class_removed:
        model = run_training(model=LeNet(), train_data=d_cr_train, test_data=d_cr_test, epochs=args.epochs)
        evaluate_log(model, d_train, d_test, d_cr_train, d_cr_test, removed_train_data=d_cr_r_train,
                     removed_test_data=d_cr_r_test, prefix="Retraining removing class")

        for e, lr, b, l in unlearn_combinations(ul_epochs, ul_learning_rates, ul_batch_sizes, ul_lambdas):
            model = unlearn(model_init, d_cr_train, d_cr_r_train, unlearn_epochs=e,
                            learning_rate=lr, batch_size=b, lambda_var=l)
            evaluate_log(model, d_train, d_test, d_cr_train, d_cr_test,
                         removed_train_data=d_cr_r_train, removed_test_data=d_cr_r_test,
                         prefix=f"Unlearning removing class (epochs={e}, lr={lr}, batch_size={b}, lambda={l})")

    if args.elements_removed:
        train_data_reduced, test_data, elements_removed, _ = get_dataloaders(batch_size=args.batch_size,
                                                                             remove_elements=args.elements)
        model = run_training(model=LeNet(), train_data=train_data_reduced, test_data=test_data, epochs=args.epochs)
        evaluate_log(model, d_train, test_data, train_data_reduced, elements_removed=elements_removed,
                     prefix="After removing elements")

        for e, lr, b, l in unlearn_combinations(ul_epochs, ul_learning_rates, ul_batch_sizes, ul_lambdas):
            model = unlearn(model_init, train_data_reduced, elements_removed, unlearn_epochs=e,
                            learning_rate=lr, batch_size=b, lambda_var=l)
            evaluate_log(model, d_train, d_test, d_cr_train, d_cr_test,
                         removed_train_data=d_cr_r_train, removed_test_data=d_cr_r_test,
                         prefix=f"Unlearning removing {args.elements_removed} elements (epochs={e}, lr={lr}, batch_size={b}, lambda={l})")


def unlearn_combinations(
        unlearn_epochs: Iterable[int],
        unlearn_learning_rates: Iterable[float],
        unlearn_batch_sizes: Iterable[int],
        unlearn_lambdas: Iterable[float],
        as_dict: bool = False
) -> Iterator[Union[Tuple[int, float, int, float], Dict[str, Union[int, float]]]]:
    """
    Gibt einen Iterator über alle Kombinationen zurück.
    - Als Tupel: (epoch, lr, batch_size, lambda)
    - Als dict (as_dict=True): {'epochs': ..., 'lr': ..., 'batch_size': ..., 'lambda': ...}
    """
    prod_iter = itertools.product(unlearn_epochs, unlearn_learning_rates,
                                  unlearn_batch_sizes, unlearn_lambdas)
    if as_dict:
        return ({'epochs': e, 'lr': lr, 'batch_size': bs, 'lambda': lam} for e, lr, bs, lam in prod_iter)
    return prod_iter


if __name__ == "__main__":
    main()
