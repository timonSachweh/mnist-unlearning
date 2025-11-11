import argparse
import re
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input log file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output file")

    args = parser.parse_args()

    data = []
    with open(args.input_file, "r") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            m = re.match(r"^-* Results for Unlearning .* \(epochs=(\d+), lr=(\d+.\d+), batch_size=(\d+), lambda=(\d+.\d+)\).*$", line)
            if m:
                e, lr, b, l = m.groups()
                train, test, train_reduced, test_reduced, train_removed, test_removed = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                i += 1
                while not re.match(r"^-*$", lines[i]):
                    line = lines[i]
                    m_train = re.match(r"^train dataset: .* acc (\d+.\d+)$", line)
                    m_test = re.match(r"^test dataset: .* acc (\d+.\d+)$", line)
                    m_train_reduced = re.match(r"^train dataset reduced: .* acc (\d+.\d+)$", line)
                    m_test_reduced = re.match(r"^test dataset reduced: .* acc (\d+.\d+)$", line)
                    m_train_removed = re.match(r"^removed train dataset: .* acc (\d+.\d+)$", line)
                    m_test_removed = re.match(r"^removed test dataset: .* acc (\d+.\d+)$", line)

                    if m_train:
                        train = float(m_train.group(1))
                    elif m_test:
                        test = float(m_test.group(1))
                    elif m_train_reduced:
                        train_reduced = float(m_train_reduced.group(1))
                    elif m_test_reduced:
                        test_reduced = float(m_test_reduced.group(1))
                    elif m_train_removed:
                        train_removed = float(m_train_removed.group(1))
                    elif m_test_removed:
                        test_removed = float(m_test_removed.group(1))
                    i += 1

                data.append({
                    "epochs": int(e),
                    "lr": float(lr),
                    "batch_size": int(b),
                    "lambda": float(l),
                    "train": train,
                    "test": test,
                    "train_reduced": train_reduced,
                    "test_reduced": test_reduced,
                    "train_removed": train_removed,
                    "test_removed": test_removed
                })

    df = pd.DataFrame(data)
    df.to_csv(args.output_file, index=False)



if __name__ == "__main__":
    main()
