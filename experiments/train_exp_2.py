from train import main, train_epoch_mixup
import torch


def run_experiment(n_runs=3):
    """Run training with Mixup augmentation."""
    # Initialize hyperparameters
    hyperparams = {
        "emb_dim": 128,
        "n_layers": 2,
        "n_heads": 8,
        "forward_dim": 256,
        "dropout": 0.15,
        "learning_rate": 2e-4,
        "batch_size": 16,
        "epochs": 2,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    size = "length"

    results = {}
    
    for run in range(n_runs):       
        seed = 42 + run
        model, accuracy, g_accuracy = main(
            train_path, test_path, size, hyperparams, oracle=True, random_seed=seed
        )

        # Collect results for display
        results[f"run_{run}"] = (accuracy, g_accuracy)

        print(f"\nRun {run}:")
        print("-" * 50)
        print(f"Greedy Accuracy on New Commands: {accuracy:.4f}")
        print(f"Oracle Accuracy on New Commands: {g_accuracy:.4f}")

    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Run | Accuracy | Greedy Accuracy")
    print("-" * 50)
    for run, (accuracy, g_accuracy) in results.items():
        print(f"{run} | {accuracy:.4f} | {g_accuracy:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    run_experiment()
