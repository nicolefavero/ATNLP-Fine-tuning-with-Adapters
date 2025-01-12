from train import main, train_epoch_mixup
from dataset import SCANDataset
import torch
import random
from torch.utils.data import DataLoader, Subset


def run_experiment(n_runs=3, dataset_fraction=0.01):
    """Run training with Mixup augmentation using a subset of the dataset."""
    # Initialize hyperparameters
    hyperparams = {
        "model_name": "t5-small",  # T5
        "max_len": 128,
        "learning_rate": 2e-4,
        "batch_size": 8,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    size = "length"

    # Load datasets
    full_train_dataset = SCANDataset(train_path, tokenizer_name="t5-small", max_len=128)
    full_test_dataset = SCANDataset(test_path, tokenizer_name="t5-small", max_len=128)

    # Create subset indices
    train_subset_size = int(len(full_train_dataset) * dataset_fraction)
    test_subset_size = int(len(full_test_dataset) * dataset_fraction)

    train_indices = random.sample(range(len(full_train_dataset)), train_subset_size)
    test_indices = random.sample(range(len(full_test_dataset)), test_subset_size)

    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_test_dataset, test_indices)

    # Replace dataset paths with dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], shuffle=False
    )

    results = {}

    for run in range(n_runs):
        seed = 42 + run

        # Use sampled dataloaders in the training process
        (
            _,  # Ignore the model object
            best_teacher_forcing_accuracy,
            final_greedy_token_acc,
            final_greedy_seq_acc,
            greedy_action_acc,
            greedy_command_acc,
            final_oracle_token_acc,
            final_oracle_seq_acc,
        ) = main(
            train_loader,
            test_loader,
            size,
            hyperparams,
            oracle=True,
            random_seed=seed,
            train_fn=train_epoch_mixup,  # Use mixup training function
        )

        # Store results for each run
        results[f"run_{run}"] = {
            "best_teacher_forcing_accuracy": best_teacher_forcing_accuracy,
            "final_greedy_token_acc": final_greedy_token_acc,
            "final_greedy_seq_acc": final_greedy_seq_acc,
            "final_oracle_token_acc": final_oracle_token_acc,
            "final_oracle_seq_acc": final_oracle_seq_acc,
        }

        # Print results for this run
        print(f"\nRun {run}:")
        print("-" * 50)
        print(f"Best Teacher Forcing Token Accuracy: {best_teacher_forcing_accuracy:.4f}")
        print(f"Final Greedy Token Accuracy: {final_greedy_token_acc:.4f}")
        print(f"Final Greedy Sequence Accuracy: {final_greedy_seq_acc:.4f}")
        print(f"Final Oracle Token Accuracy: {final_oracle_token_acc:.4f}")
        print(f"Final Oracle Sequence Accuracy: {final_oracle_seq_acc:.4f}")
        print(f"Greedy Accuracy by Action Length: {greedy_action_acc}")
        print(f"Greedy Accuracy by Command Length: {greedy_command_acc}")

    # Summary of all runs
    print("\nFinal Results Summary:")
    print("=" * 70)
    print(
        "Run | Teacher Forcing | Greedy Token | Greedy Sequence | Oracle Token | Oracle Sequence"
    )
    print("-" * 70)
    for run, metrics in results.items():
        print(
            f"{run} | {metrics['best_teacher_forcing_accuracy']:.4f} | "
            f"{metrics['final_greedy_token_acc']:.4f} | {metrics['final_greedy_seq_acc']:.4f} | "
            f"{metrics['final_oracle_token_acc']:.4f} | {metrics['final_oracle_seq_acc']:.4f}"
        )
    print("-" * 70)


if __name__ == "__main__":
    run_experiment()
