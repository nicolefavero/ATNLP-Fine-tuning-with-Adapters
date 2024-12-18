import numpy as np
from train import main
import torch


def get_add_prim_dataset_pairs():
    """Get pairs of training and test dataset paths for Experiment 3."""
    base_path = "data/add_prim_split"

    # Add the num0 dataset explicitly
    pairs = [
        (
            f"{base_path}/tasks_train_addprim_jump.txt",
            f"{base_path}/tasks_test_addprim_jump.txt",
            "jump",
        ),
        (
            f"{base_path}/tasks_train_addprim_turn_left.txt",
            f"{base_path}/tasks_test_addprim_turn_left.txt",
            "turn_left",
        ),
    ]

    additional_base_path = "data/add_prim_split/with_additional_examples"
    num_composed_commands = ["num1", "num2", "num4", "num8", "num16", "num32"]
    for num in num_composed_commands:
        for rep in range(1, 2):  # Assuming 5 repetitions for each split
            train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}_rep{rep}.txt"
            test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}_rep{rep}.txt"
            pairs.append((train_path, test_path, f"{num}_rep{rep}"))

    return pairs


def run_experiment_3(n_runs=1):
    """
    Run Experiment 3: Adding a new primitive and testing generalization to composed commands.
    This function trains and evaluates the Transformer model for each dataset variation
    (e.g., num0, num1, ..., num32), iterating over the specified dataset splits.
    """
    n_runs = n_runs
    results = {}

    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # Fetch dataset pairs
    pairs = get_add_prim_dataset_pairs()

    # Run training and evaluation for each dataset pair
    for train_path, test_path, size in pairs:
        results[size] = []
        print(f"\nStarting training for dataset {size}")
        print("=" * 70)

        for run in range(n_runs):
            seed = 42 + run  # Different seed for each run
            print(f"Run {run+1}/{n_runs} with seed {seed}")

            # Call the existing `main` function with hyperparameters
            _, accuracy, g_accuracy = main(
                train_path, test_path, size, hyperparams, random_seed=seed, oracle=False
            )
            results[size].append((accuracy, g_accuracy))
            print(f"Run {run+1} Accuracy: {accuracy:.4f}, Greedy Accuracy: {g_accuracy:.4f}")

    # Print summary of results
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)

    for size, accuracies in results.items():
        accuracies = [(acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                   g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc) 
                  for acc, g_acc in accuracies]
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)
        print(f"{size:11} | Mean Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")
        print("-" * 50)


if __name__ == "__main__":
    run_experiment_3()
