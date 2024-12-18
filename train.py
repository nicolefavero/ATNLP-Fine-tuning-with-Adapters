import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from dataset import SCANDataset
from model.transformer import Transformer
from tqdm import tqdm
import numpy as np
from torchmetrics import Accuracy
from utils.utils import SequenceAccuracy, greedy_decode, oracle_greedy_search
from typing import Tuple, Callable
from torch.utils.tensorboard import SummaryWriter
torch.set_float32_matmul_precision('high')

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

        output = output.reshape(-1, output.shape[-1])
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
            encode_out = model.encoder(src, model.create_src_mask(src))
            pred = torch.full(
                (src.size(0), 1), model.tgt_pad_idx, dtype=torch.long, device=device
            )

            all_logits = []
            max_len = tgt_output.size(1)

            for _ in range(max_len):
                tgt_mask = model.create_tgt_mask(pred)
                decode_out = model.decoder(
                    pred, encode_out, model.create_src_mask(src), tgt_mask
                )

                last_step_logits = decode_out[:, -1, :]
                all_logits.append(last_step_logits)

                next_tokens = torch.argmax(last_step_logits, dim=-1)
                next_tokens = next_tokens.unsqueeze(1)
                pred = torch.cat([pred, next_tokens], dim=1)

            logits = torch.stack(all_logits, dim=1)
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
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model using greedy search decoding.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
    """
    model.eval()
    total_loss = 0

    tgt_eos_idx = dataloader.dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataloader.dataset.tgt_vocab.tok2id["<BOS>"]
    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    accuracy = Accuracy(
        ignore_index=tgt_pad_idx,
        task="multiclass",
        num_classes=dataloader.dataset.tgt_vocab.vocab_size,
    ).to(device)
    seq_accuracy = SequenceAccuracy(
        tgt_pad_idx=tgt_pad_idx, tgt_eos_idx=tgt_eos_idx
    ).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]

            # Use greedy decoding
            output = greedy_decode(
                model, src, tgt_eos_idx, tgt_bos_idx, device, return_logits=True
            )
            flattened_output = output.reshape(-1, output.shape[-1])
            flattened_tgt = tgt_output.reshape(-1)

            accuracy.update(flattened_output.argmax(dim=-1), flattened_tgt)
            pred_sequences = output.argmax(dim=-1)
            seq_accuracy.update(pred_sequences, tgt_output)

            loss = criterion(flattened_output, flattened_tgt)
            total_loss += loss.item()
    acc = accuracy.compute()
    seq_acc = seq_accuracy.compute()
    return total_loss / len(dataloader), acc, seq_acc


def evaluate_teacher_forcing(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model using teacher forcing.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
    """
    model.eval()
    total_loss = 0

    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    tgt_eos_idx = dataloader.dataset.tgt_vocab.tok2id["<EOS>"]
    accuracy = Accuracy(
        ignore_index=tgt_pad_idx,
        task="multiclass",
        num_classes=dataloader.dataset.tgt_vocab.vocab_size,
    ).to(device)
    seq_accuracy = SequenceAccuracy(
        tgt_pad_idx=tgt_pad_idx, tgt_eos_idx=tgt_eos_idx
    ).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Teacher Forcing"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Directly pass src and tgt_input to the model
            output = model(src, tgt_input)
            flattened_output = output.reshape(-1, output.shape[-1])
            flattened_tgt = tgt_output.reshape(-1)
            accuracy.update(flattened_output.argmax(dim=-1), flattened_tgt)

            pred_sequences = output.argmax(dim=-1)
            seq_accuracy.update(pred_sequences, tgt_output)

            loss = criterion(flattened_output, flattened_tgt)
            total_loss += loss.item()
    acc = accuracy.compute()
    seq_acc = seq_accuracy.compute()

    return total_loss / len(dataloader), acc, seq_acc


def evaluate_oracle_greedy_search(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Evaluates the model using oracle greedy search decoding.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for the evaluation data.
        criterion: Loss function.
        device: Device to run the evaluation on.
    """
    model.eval()
    total_loss = 0

    tgt_eos_idx = dataloader.dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataloader.dataset.tgt_vocab.tok2id["<BOS>"]
    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    accuracy = Accuracy(
        ignore_index=tgt_pad_idx,
        task="multiclass",
        num_classes=dataloader.dataset.tgt_vocab.vocab_size,
    ).to(device)
    seq_accuracy = SequenceAccuracy(
        tgt_pad_idx=tgt_pad_idx, tgt_eos_idx=tgt_eos_idx
    ).to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Oracle Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]
            
            # Use oracle greedy decoding
            output = oracle_greedy_search(
                model,
                src,
                tgt_eos_idx,
                tgt_bos_idx,
                tgt_output,
                device,
                return_logits=True,
            )
            flattened_output = output.reshape(-1, output.shape[-1])
            flattened_tgt = tgt_output.reshape(-1)
            accuracy.update(flattened_output.argmax(dim=-1), flattened_tgt)

            pred_sequences = output.argmax(dim=-1)
            seq_accuracy.update(pred_sequences, tgt_output)

            loss = criterion(flattened_output, flattened_tgt)
            total_loss += loss.item()
    acc = accuracy.compute()
    seq_acc = seq_accuracy.compute()
    return total_loss / len(dataloader), acc, seq_acc


def main(
    train_path: str,
    test_path: str,
    model_suffix: str,
    hyperparams: dict,
    random_seed: int = 42,
    oracle: bool = False,
    train_fn: Callable = train_epoch_teacher_forcing,
) -> Tuple[Transformer, float]:
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
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    EMB_DIM = hyperparams["emb_dim"]
    N_LAYERS = hyperparams["n_layers"]
    N_HEADS = hyperparams["n_heads"]
    FORWARD_DIM = hyperparams["forward_dim"]
    DROPOUT = hyperparams["dropout"]
    LEARNING_RATE = hyperparams["learning_rate"]
    BATCH_SIZE = hyperparams["batch_size"]
    EPOCHS = hyperparams["epochs"]
    DEVICE = hyperparams["device"]

    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True
    )

    model = Transformer(
        src_vocab_size=dataset.src_vocab.vocab_size,
        tgt_vocab_size=dataset.tgt_vocab.vocab_size,
        src_pad_idx=dataset.src_vocab.tok2id["<PAD>"],
        tgt_pad_idx=dataset.tgt_vocab.tok2id["<PAD>"],
        emb_dim=EMB_DIM,
        num_layers=N_LAYERS,
        num_heads=N_HEADS,
        forward_dim=FORWARD_DIM,
        dropout=DROPOUT,
        max_len=dataset.max_len,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab.tok2id["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    writer = SummaryWriter(log_dir=f"runs/{model_suffix}")

    model.compile(mode="reduce-overhead", fullgraph=True)

    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, accuracy, seq_acc = evaluate_teacher_forcing(
            model, test_loader, criterion, DEVICE
        )
        # g_test_loss, g_accuracy, g_seq_acc = evaluate_greedy_search(
        #     model, test_loader, criterion, DEVICE
        # )
        if oracle:
            go_test_loss, go_accuracy, go_seq_acc = evaluate_oracle_greedy_search(
                model, test_loader, criterion, DEVICE
            )

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Test_Token', accuracy, epoch)
        writer.add_scalar('Accuracy/Test_Sequence', seq_acc, epoch)
        # writer.add_scalar('Accuracy/Greedy_Token', g_accuracy, epoch)
        # writer.add_scalar('Accuracy/Greedy_Sequence', g_seq_acc, epoch)
        if oracle:
            writer.add_scalar('Accuracy/Oracle_Greedy_Token', go_accuracy, epoch)
            writer.add_scalar('Accuracy/Oracle_Greedy_Sequence', go_seq_acc, epoch)

        print(f"Dataset {model_suffix} - Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        # print(f"Greedy Search Loss: {g_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Sequence Length Accuracy: {seq_acc:.4f}")
        # print(
        #     f"Greedy Search Accuracy: {g_accuracy:.4f}, Sequence Length Accuracy: {g_seq_acc:.4f}"
        # )
        if oracle:
            print(f"Oracle Greedy Search Loss: {go_test_loss:.4f}")
            print(
                f"Oracle Greedy Search Accuracy: {go_accuracy:.4f}, Sequence Length Accuracy: {go_seq_acc:.4f}"
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                },
                f"model/best_model_{model_suffix}.pt",
            )

        print("-" * 50)

    writer.close()
    print(f"Training completed for p{model_suffix}. Best accuracy: {best_accuracy:.4f}")

    # Load the best model
    checkpoint = torch.load(f"model/best_model_{model_suffix}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate the best model on greedy search
    _, g_accuracy, seq_acc = evaluate_greedy_search(model, test_loader, criterion, DEVICE)
    print(f"Greedy Search Evaluation - Accuracy: {g_accuracy:.4f}, Sequence Length Accuracy: {seq_acc:.4f}")

    return model, best_accuracy, g_accuracy