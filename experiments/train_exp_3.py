import numpy as np
from train import main
import torch
from rich import print
from rich.traceback import install
from model.t5_transformer import T5Wrapper

install()


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
        train_test_pairs = []
        for rep in range(1, 6):  # Changed to 5 repetitions
            train_path = f"{additional_base_path}/tasks_train_addprim_complex_jump_{num}_rep{rep}.txt"
            test_path = f"{additional_base_path}/tasks_test_addprim_complex_jump_{num}_rep{rep}.txt"
            train_test_pairs.append((train_path, test_path))
        pairs.append((train_test_pairs, num))

    return pairs


def run_experiment_3(n_runs=5):
    """
    Run Experiment 3: Adding a new primitive and testing generalization to composed commands.
    Uses n_runs for basic cases (jump, turn_left) and existing 5 repetitions for numerical cases.
    """
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

    # Process the basic jump and turn_left cases with n_runs
    for train_path, test_path, name in pairs[:2]:
        print(f"\nProcessing dataset {name}")
        print("=" * 70)

        basic_results = []
        for run in range(n_runs):
            seed = 42 + run
            print(f"Run {run+1}/{n_runs} with seed {seed}")
            _, accuracy, g_accuracy, seq_accuracy, *_ = main(
                train_path, test_path, name, hyperparams, random_seed=seed, oracle=False
            )
            basic_results.append((accuracy, g_accuracy, seq_accuracy))
        results[name] = basic_results

    # Process the numerical cases (using existing 5 repetitions)
    for train_test_pairs, num in pairs[2:]:
        print(f"\nProcessing dataset {num}")
        print("=" * 70)

        rep_results = []
        for train_path, test_path in train_test_pairs:
            _, accuracy, g_accuracy, seq_accuracy, *_= main(
                train_path, test_path, num, hyperparams, random_seed=42, oracle=False
            )
            rep_results.append((accuracy, g_accuracy, seq_accuracy))
        results[num] = rep_results

    # Print summary of results
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev | Mean Greedy Accuracy ± Std Dev | Mean Sequence Accuracy ± Std Dev")
    print("-" * 120)


    for size, accuracies in results.items():
        accuracies = [
            (
                acc.cpu().numpy() if torch.is_tensor(acc) else acc,
                g_acc.cpu().numpy() if torch.is_tensor(g_acc) else g_acc,
                s_acc.cpu().numpy() if torch.is_tensor(s_acc) else s_acc,
            )
            for acc, g_acc, s_acc in accuracies
        ]
        mean = np.mean(accuracies, axis=0)
        std = np.std(accuracies, axis=0)
        print(f"{size:11} | Mean Accuracy: {mean[0]:.4f} ± {std[0]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[0]:.4f}' for acc in accuracies)}")
        print(f"Mean Greedy Accuracy: {mean[1]:.4f} ± {std[1]:.4f}")
        print(f"Individual runs: {', '.join(f'{acc[1]:.4f}' for acc in accuracies)}\n")
        print(f"Mean Sequence Accuracy: {mean[2]:.4f} ± {std[2]:.4f}")
        print(f"Individual runs: [{', '.join(f'{acc[2]:.4f}' for acc in accuracies)}]\n")
        print("-" * 50)


#if __name__ == "__main__":
#    run_experiment_3()

if __name__ == "__main__":
    # Test with just one primitive case and fewer runs
    hyperparams = {
        "model_name": "t5-small",  # Start with the smallest T5 model
        "learning_rate": 1e-4,
        "batch_size": 8,          # Smaller batch size for testing
        "epochs": 2,              # Fewer epochs for testing
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    # Get just the first test case (jump)
    base_path = "data/add_prim_split"
    train_path = f"{base_path}/tasks_train_addprim_jump.txt"
    test_path = f"{base_path}/tasks_test_addprim_jump.txt"
    
    print("Testing T5 model with jump dataset...")
    _, accuracy, g_accuracy, seq_accuracy, *_ = main(
        train_path,
        test_path,
        "t5_test",
        hyperparams,
        random_seed=42,
        oracle=False,
        model_class=T5Wrapper
    )
    
    print(f"\nResults:")
    print(f"Token Accuracy: {accuracy:.4f}")
    print(f"Greedy Accuracy: {g_accuracy:.4f}")
    print(f"Sequence Accuracy: {seq_accuracy:.4f}")