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

def train_epoch_mixup(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """
    Trains the model for one epoch using mixed teacher forcing and scheduled sampling.

    Args:
        model: The model to train (T5Wrapper).
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to run the training on.
        teacher_forcing_ratio: Ratio of teacher forcing. Defaults to 0.5.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training with Mixup"):
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        optimizer.zero_grad()

        if random.random() < teacher_forcing_ratio:
            # Teacher forcing
            loss, _ = model(src, tgt)
        else:
            # Scheduled sampling with greedy decoding
            batch_size = src.size(0)
            max_len = model.max_len

            pred = torch.full(
                (batch_size, 1),
                model.tokenizer.bos_token_id,
                dtype=torch.long,
                device=device,
            )
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            all_logits = []

            for _ in range(max_len - 1):
                if torch.all(finished):
                    break

                outputs = model.model(
                    input_ids=src,
                    decoder_input_ids=pred,
                    return_dict=True,
                )
                logits_step = outputs.logits[:, -1, :]  # Only last time step
                all_logits.append(logits_step.unsqueeze(1))

                next_tokens = torch.argmax(logits_step, dim=-1).unsqueeze(1)
                pred = torch.cat([pred, next_tokens], dim=1)
                finished = finished | (next_tokens.squeeze(1) == model.tokenizer.eos_token_id)

            # Concatenate logits and reshape for loss calculation
            logits = torch.cat(all_logits, dim=1)  # Shape: (batch_size, seq_len, vocab_size)
            if logits.size(1) > tgt[:, 1:].size(1):
                logits = logits[:, :tgt[:, 1:].size(1), :]  # Trim logits to match target length
            elif logits.size(1) < tgt[:, 1:].size(1):
                tgt = tgt[:, :logits.size(1) + 1]  # Trim target to match logits length

            logits = logits.reshape(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            tgt_output = tgt[:, 1:].reshape(-1)  # Shape: (batch_size * seq_len)

            loss = criterion(logits, tgt_output)

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

            outputs = model.model.generate(input_ids=src, max_length=model.max_len)
            pred_ids = outputs[:, 1:]  # Skip BOS token for predictions
            tgt_output = tgt[:, 1:]  # Skip BOS token for targets

            # Calculate loss
            logits = model.model(
                input_ids=src, decoder_input_ids=tgt[:, :-1], return_dict=True
            ).logits
            logits = logits.reshape(-1, logits.size(-1))
            tgt_output_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(logits, tgt_output_flat)
            total_loss += loss.item()

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
    train_loader,
    test_loader,
    size: str,
    hyperparams: dict,
    oracle: bool = False,
    random_seed: int = 42,
    train_fn=train_epoch_mixup,
):
    def set_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

    set_seed(random_seed)

    LEARNING_RATE = hyperparams["learning_rate"]
    DEVICE = hyperparams["device"]
    EPOCHS = hyperparams["epochs"]

    # Initialize model
    model = T5Wrapper(model_name=hyperparams["model_name"], max_len=hyperparams["max_len"]).to(DEVICE)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=f"runs/{size}")

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
            torch.save(model.state_dict(), f"model/best_model_{size}.pt")

    writer.close()

    model.load_state_dict(torch.load(f"model/best_model_{size}.pt"))

    return model, best_accuracy, token_acc, seq_acc, {}, {}, 0.0, 0.0
