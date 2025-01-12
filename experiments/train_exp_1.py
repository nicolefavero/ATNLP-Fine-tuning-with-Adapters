from train import main
from dataset import SCANDataset
import torch
import numpy as np
from rich import print
from rich.traceback import install

install()


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def run_all_variations(n_runs=1):
    """Run training 5 times for all dataset size variations with different seeds."""
    results = {}

    # Initialize hyperparameters
    hyperparams = {
        "model_name": "t5-small",  # T5
        "learning_rate": 7e-4,
        "batch_size": 64,
        "epochs": 20,
        "max_len": 128,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    for train_path, test_path, size in get_dataset_pairs():
        results[f"p{size}"] = []

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")
        print("=" * 70)

        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            _, accuracy, g_accuracy, *_ = main(
                train_path,
                test_path,
                f"p_{size}",
                hyperparams,
                random_seed=seed,
            )
            results[f"p{size}"].append((accuracy, g_accuracy))

    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)
        print(f"{size:11} | Mean Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")


if __name__ == "__main__":
    run_all_variations()