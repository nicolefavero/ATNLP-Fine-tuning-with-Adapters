from train import main, train_epoch_mixup
from model.t5_transformer import T5Wrapper, T5Config
import torch
import numpy as np
from rich import print
from rich.traceback import install

install()

def run_p32(n_runs=1):
    """Run training n_runs times for the p_32 dataset with different seeds."""
    results = []

    # Paths for p32
    train_path = "data/simple_split/size_variations/tasks_train_simple_p32.txt"
    test_path = "data/simple_split/size_variations/tasks_test_simple_p32.txt"
    size_str = "p32"

    # Initialize hyperparameters
    hyperparams = {
        "model_name": "google-t5/t5-small",  # example model name
        "learning_rate": 1e-4,
        "batch_size": 128,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "mps"),
    }

    for run in range(n_runs):
        seed = 42 + run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")
        print("=" * 70)

        _, accuracy, g_accuracy, *_ = main(
            train_path,
            test_path,
            size_str,
            hyperparams,
            random_seed=seed,
            model_class=T5Wrapper,  # specify T5Wrapper as the model class
            train_fn=train_epoch_mixup
        )
        results.append((accuracy, g_accuracy))

    # Final results summary
    print("\nFinal Results Summary for p32:")
    print("=" * 50)
    print("Mean Accuracy ± Std Dev (Teacher Forcing) | Mean Accuracy ± Std Dev (Greedy)")

    # Convert any tensor values to numpy
    accuracies = [
        (
            acc.cpu().numpy() if torch.is_tensor(acc) else acc,
            g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
        )
        for acc, g_acc in results
    ]
    mean = np.mean(accuracies, axis=0)
    std = np.std(accuracies, axis=0)

    print(f"Teacher-Forcing Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
    print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
    print(f"Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
    print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")

if __name__ == "__main__":
    run_p32()
