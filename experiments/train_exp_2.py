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
        (
            _,  # Ignore the model object
            best_teacher_forcing_accuracy,  # Best token accuracy (teacher forcing)
            final_greedy_token_acc,  # Final token accuracy (greedy search)
            final_greedy_seq_acc,  # Final sequence accuracy (greedy search)
            greedy_action_acc,  # Greedy accuracy grouped by action length
            greedy_command_acc,  # Greedy accuracy grouped by command length
            final_oracle_token_acc,  # Final token accuracy (oracle search)
            final_oracle_seq_acc,  # Final sequence accuracy (oracle search)
        ) = main(train_path, test_path, size, hyperparams, oracle=True, random_seed=seed)

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
