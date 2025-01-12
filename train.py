import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SCANDataset
from model.transformer import T5Wrapper
from tqdm import tqdm
import random
from rich import print
from rich.traceback import install
from utils.utils import greedy_decode, oracle_greedy_search, calculate_accuracy
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

install()

GRAD_CLIP = 1

def train_epoch_teacher_forcing(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Trains the model for one epoch using teacher forcing.

    Args:
        model: The model to train.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to run the training on.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(src, tgt)
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate_greedy_search(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    eos_idx: int,
    pad_idx: int,
) -> Tuple[float, float, float, dict, dict]:
    """
    Evaluates the model using greedy search decoding.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
        eos_idx: End of sequence token index.
        pad_idx: Padding token index.

    Returns:
        - Average loss
        - Token-level accuracy
        - Sequence-level accuracy
        - Length-specific accuracies by action length
        - Length-specific accuracies by command length
    """
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Greedy Search"):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            loss, _ = model(src, tgt)
            total_loss += loss.item()

            pred_ids = model(src)
            pred_ids = pred_ids[:, 1:]  # Skip BOS token for predictions
            tgt_output = tgt[:, 1:]  # Skip BOS token for targets

            for i in range(len(pred_ids)):
                token_acc, seq_acc = calculate_accuracy(
                    pred_ids[i].unsqueeze(0), tgt_output[i].unsqueeze(0), pad_idx, eos_idx
                )
                token_accuracies.append(token_acc)
                seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc, {}, {}

def main(
    train_path: str,
    test_path: str,
    model_suffix: str,
    hyperparams: dict,
    random_seed: int = 42,
    oracle: bool = False,
    train_fn: Callable = train_epoch_teacher_forcing,
) -> Tuple[T5Wrapper, float, float, float, dict, dict, float, float]:
    """
    Main function to train and evaluate the model on the SCAN dataset.
    """
    def set_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

    set_seed(random_seed)

    LEARNING_RATE = hyperparams["learning_rate"]
    BATCH_SIZE = hyperparams["batch_size"]
    EPOCHS = hyperparams["epochs"]
    DEVICE = hyperparams["device"]

    train_dataset = SCANDataset(train_path, tokenizer_name="t5-small", max_len=128)
    test_dataset = SCANDataset(test_path, tokenizer_name="t5-small", max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = T5Wrapper(model_name="t5-small", max_len=128).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=f"runs/{model_suffix}")

    best_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, token_acc, seq_acc, _, _ = evaluate_greedy_search(
            model, test_loader, criterion, DEVICE, model.tokenizer.eos_token_id, model.tokenizer.pad_token_id
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Token", token_acc, epoch)
        writer.add_scalar("Accuracy/Sequence", seq_acc, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Token Accuracy: {token_acc:.4f}, Sequence Accuracy: {seq_acc:.4f}")

        if token_acc > best_accuracy:
            best_accuracy = token_acc
            torch.save(model.state_dict(), f"model/best_model_{model_suffix}.pt")

    writer.close()

    model.load_state_dict(torch.load(f"model/best_model_{model_suffix}.pt"))

    return model, best_accuracy, token_acc, seq_acc, {}, {}, 0.0, 0.0
