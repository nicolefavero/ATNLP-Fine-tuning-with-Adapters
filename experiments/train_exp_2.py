from train import main, train_epoch_mixup
import torch


def run_experiment():
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
        "epochs": 20,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    train_path = "data/length_split/tasks_train_length.txt"
    test_path = "data/length_split/tasks_test_length.txt"
    size = "length"

    # Run training with Mixup augmentation
    _, accuracy = main(
        train_path, test_path, size, hyperparams, train_fn=train_epoch_mixup
    )
    print(f"Accuracy: {accuracy:.4f}")
