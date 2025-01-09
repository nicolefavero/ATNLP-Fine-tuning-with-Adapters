import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import SCANDataset
from model.transformer import Transformer
from tqdm import tqdm
import random
from rich import print
from rich.traceback import install

install()
import numpy as np
from utils.utils import greedy_decode, oracle_greedy_search, calculate_accuracy
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

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
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output = model(src, tgt_input)

        output = output.view(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
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
        model: The model to train.
        dataloader: DataLoader for the training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        device: Device to run the training on.
        teacher_forcing_ratio: Ratio of teacher forcing. Defaults to 0.5.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()

        if np.random.rand() < teacher_forcing_ratio:
            output = model(src, tgt_input)
        else:
            batch_size = src.size(0)
            max_len = model.max_len
            encode_out = model.encoder(src, model.create_src_mask(src))
            pred = torch.full(
                (batch_size, 1),
                dataloader.dataset.tgt_vocab.tok2id["<BOS>"],  # Changed to tgt_vocab
                dtype=torch.long,
                device=device,
            )

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            all_logits = []

            for _ in range(max_len - 1):
                if torch.all(finished):
                    break

                tgt_mask = model.create_tgt_mask(pred)
                decode_out = model.decoder(
                    pred, encode_out, model.create_src_mask(src), tgt_mask
                )

                last_step_logits = decode_out[:, -1, :]
                all_logits.append(last_step_logits)

                next_tokens = torch.argmax(last_step_logits, dim=-1)
                next_tokens = next_tokens.unsqueeze(1)
                pred = torch.cat([pred, next_tokens * ~finished.unsqueeze(1)], dim=1)

                newly_finished = (
                    next_tokens == dataloader.dataset.vocab.tok2id["<EOS>"]
                ).squeeze(1)
                finished = finished | newly_finished
                logits = torch.stack(all_logits, dim=1)
                if logits.size(1) < max_len - 1:
                    pad_size = (max_len - 1) - logits.size(1)
                    logits = F.pad(logits, (0, 0, 0, pad_size))
                else:
                    logits = logits[:, : (max_len - 1), :]

            output = logits

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
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
    bos_idx: int,
    pad_idx: int,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model using greedy search decoding.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
        eos_idx: End of sequence token index.
        bos_idx: Beginning of sequence token index.
        pad_idx: Padding token index.
    """
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]

            # Use greedy decoding
            pred = greedy_decode(
                model,
                src,
                eos_idx,
                bos_idx,
                pad_idx,
                device,
                return_logits=False,
            )
            pred = pred[:, 1:]
            token_acc, seq_acc = calculate_accuracy(pred, tgt_output, pad_idx, eos_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(dataloader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc


def evaluate_teacher_forcing(
    model, data_loader, criterion, device, pad_idx: int, eos_idx: int
):
    """Evaluate model using teacher forcing"""
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            output = output.contiguous().view(-1, output.shape[-1])
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

            # Calculate accuracies
            pred = output.argmax(dim=-1).view(tgt.size(0), -1)
            token_acc, seq_acc = calculate_accuracy(
                pred,
                tgt[:, 1:],
                pad_idx,
                eos_idx=eos_idx,
            )
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(data_loader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc


def evaluate_oracle_greedy_search(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    eos_idx: int,
    bos_idx: int,
    pad_idx: int,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model using oracle greedy search decoding.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
        eos_idx: End of sequence token index.
        bos_idx: Beginning of sequence token index.
        pad_idx: Padding token index.
    """
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Oracle Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]

            # Use oracle greedy decoding
            pred = oracle_greedy_search(
                model,
                src,
                eos_idx,
                bos_idx,
                pad_idx,
                tgt_output,
                device,
                return_logits=False,
            )
            pred = pred[:, 1:]
            token_acc, seq_acc = calculate_accuracy(pred, tgt_output, pad_idx, eos_idx)
            token_accuracies.append(token_acc)
            seq_accuracies.append(seq_acc)

    avg_loss = total_loss / len(dataloader)
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc


def main(
    train_path: str,
    test_path: str,
    model_suffix: str,
    hyperparams: dict,
    random_seed: int = 42,
    oracle: bool = False,
    train_fn: Callable = train_epoch_teacher_forcing,
) -> Tuple[Transformer, float, float, float]:
    """
    Main function to train and evaluate the model on the SCAN dataset.

    Args:
        train_path: Path to the training dataset.
        test_path: Path to the test dataset.
        model_suffix: Suffix for the model filename.
        hyperparams: Dictionary of hyperparameters.
        random_seed: Random seed for reproducibility. Defaults to 42.
        oracle: Whether to use oracle greedy search during evaluation. Defaults to False.
        train_fn: Training function to use. Defaults to train_epoch_teacher_forcing.
    """
    def set_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

    set_seed(random_seed)

    EMB_DIM = hyperparams["emb_dim"]
    N_LAYERS = hyperparams["n_layers"]
    N_HEADS = hyperparams["n_heads"]
    FORWARD_DIM = hyperparams["forward_dim"]
    DROPOUT = hyperparams["dropout"]
    LEARNING_RATE = hyperparams["learning_rate"]
    BATCH_SIZE = hyperparams["batch_size"]
    DEVICE = hyperparams["device"]

    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True,
    )

    # Get special tokens for source and target vocabularies
    src_pad_idx = dataset.src_vocab.tok2id["<PAD>"]
    
    tgt_pad_idx = dataset.tgt_vocab.tok2id["<PAD>"]
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataset.tgt_vocab.tok2id["<BOS>"]

    # Dynamic epochs based on dataset size
    data_len = dataset.__len__()
    if (100000 // data_len) > 100:
        hyperparams["epochs"] = (100000 // data_len) 
    else:
        hyperparams["epochs"] = min(20, (100000 // data_len))

    EPOCHS = hyperparams["epochs"]

    # Initialize model
    model = Transformer(
        src_vocab_size=dataset.src_vocab.vocab_size,
        tgt_vocab_size=dataset.tgt_vocab.vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        emb_dim=EMB_DIM,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        forward_dim=FORWARD_DIM,
        dropout=DROPOUT,
        max_len=dataset.max_len,
    ).to(DEVICE)

    # Initialize optimizer, loss function and Tensorboard writer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=f"runs/{model_suffix}")

    best_accuracy = 0.0
    lowest_loss = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, DEVICE)

        # Teacher forcing evaluation
        test_loss, tf_token_acc, tf_seq_acc = evaluate_teacher_forcing(
            model, test_loader, criterion, DEVICE, tgt_pad_idx, tgt_eos_idx
        )

        # Generation evaluation
        # gen_loss, gen_token_acc, gen_seq_acc = evaluate_greedy_search(
        #     model,
        #     test_loader,
        #     criterion,
        #     DEVICE,
        #     eos_idx,
        #     bos_idx,
        #     pad_idx,
        # )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/TeacherForcing_Token", tf_token_acc, epoch)
        writer.add_scalar("Accuracy/TeacherForcing_Sequence", tf_seq_acc, epoch)
        # writer.add_scalar('Accuracy/Generation_Token', gen_token_acc, epoch)
        # writer.add_scalar('Accuracy/Generation_Sequence', gen_seq_acc, epoch)

        print(f"Dataset {model_suffix} - Epoch: {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(
            f"Teacher Forcing - Token Acc: {tf_token_acc:.4f}, Seq Acc: {tf_seq_acc:.4f}"
        )
        # print(f"Generation - Token Acc: {gen_token_acc:.4f}, Seq Acc: {gen_seq_acc:.4f}")

        if tf_token_acc > best_accuracy:
            best_accuracy = tf_token_acc
            print(f"New best accuracy: {best_accuracy:.4f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": tf_token_acc,
                },
                f"model/best_model_{model_suffix}.pt",
            )

        print("-" * 50)

    writer.close()
    print(f"Training completed for {model_suffix}. Best accuracy: {best_accuracy:.4f}")

    # Load the best model
    checkpoint = torch.load(f"model/best_model_{model_suffix}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the best model
    _, final_token_acc, final_seq_acc = evaluate_greedy_search(
        model, test_loader, criterion, DEVICE, tgt_eos_idx, tgt_bos_idx, tgt_pad_idx
    )
    print(
        f"Final Evaluation - Token Accuracy: {final_token_acc:.4f}, Sequence Accuracy: {final_seq_acc:.4f}"
    )
    if oracle:
        _, final_token_acc, final_seq_acc = evaluate_oracle_greedy_search(
            model, test_loader, criterion, DEVICE, tgt_eos_idx, tgt_bos_idx, tgt_pad_idx
        )
        print(
            f"Final Oracle Evaluation - Token Accuracy: {final_token_acc:.4f}, Sequence Accuracy: {final_seq_acc:.4f}"
        )

    return model, best_accuracy, final_token_acc, final_seq_acc
