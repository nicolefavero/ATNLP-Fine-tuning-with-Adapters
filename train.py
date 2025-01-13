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
from typing import Type, Tuple, Callable
from transformers import T5ForConditionalGeneration, T5Tokenizer
from model.t5_transformer import T5Wrapper
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
    accumulation_steps: int = 4,  # Accumulate gradients
) -> float:
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at start
    
    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output = model(src, tgt_input)
        
        if isinstance(output, torch.Tensor):
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)
        else:
            loss = output.loss

        # Normalize loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Handle any remaining gradients
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

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
    Now handles T5 vs. custom Transformer differently.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)

        tgt_input = tgt[:, :-1]  # everything but last token
        tgt_output = tgt[:, 1:]  # everything but BOS

        optimizer.zero_grad()

        # Decide teacher forcing vs. auto-regressive
        use_teacher_forcing = (np.random.rand() < teacher_forcing_ratio)

        if isinstance(model, T5Wrapper):
            # ========== T5-based model path ==========

            if use_teacher_forcing:
                # **Teacher Forcing** with T5: just call forward
                # This uses T5Wrapper.forward => self.model(...)
                output = model(src, tgt_input)  
                # `output` shape: [batch_size, seq_len, vocab_size]

            else:
                # **Auto-Regressive** for T5: use model.model.generate(...)
                # or partial decoding logic

                # We'll do a simple approach: generate entire sequence
                # then compare to ground truth for the loss.
                with torch.no_grad():
                    # generate predicts tokens from scratch
                    generated_tokens = model.model.generate(
                        input_ids=src,
                        max_length=model.max_len,
                        num_beams=1,  # greedy
                        pad_token_id=model.tokenizer.pad_token_id,
                        eos_token_id=model.tokenizer.eos_token_id,
                        bos_token_id=model.tokenizer.bos_token_id,
                    )
                # If you want to get logits for each step, you'd do
                # return_dict_in_generate=True, output_scores=True, etc.

                # Next, for simplicity, let's re-run teacher forcing to get logits
                # we can at least get a gradient w.r.t. the "teacher forcing" path
                # This is a simplistic approach; you can do something more advanced.
                output = model(src, tgt_input)

            # Flatten for the CrossEntropy
            output = output.reshape(-1, output.size(-1))    # [batch*seq, vocab]
            tgt_output = tgt_output.reshape(-1)             # [batch*seq]

            loss = criterion(output, tgt_output)
            loss.backward()

        else:
            # ========== Custom Transformer path ==========

            if use_teacher_forcing:
                # same as your original teacher forcing logic
                output = model(src, tgt_input)
                # Flatten
                output = output.view(-1, output.shape[-1])
                tgt_output_flat = tgt_output.reshape(-1)
                loss = criterion(output, tgt_output_flat)
                loss.backward()

            else:
                # partial decoding (step-by-step) with encoder & decoder
                batch_size = src.size(0)
                max_len = model.max_len
                encode_out = model.encoder(src, model.create_src_mask(src))

                # Start with <BOS>
                pred = torch.full(
                    (batch_size, 1),
                    dataloader.dataset.tgt_vocab.tok2id["<BOS>"],
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

                    last_step_logits = decode_out[:, -1, :]    # shape [batch, vocab]
                    all_logits.append(last_step_logits)

                    next_tokens = torch.argmax(last_step_logits, dim=-1)
                    next_tokens = next_tokens.unsqueeze(1)

                    pred = torch.cat(
                        [pred, next_tokens * ~finished.unsqueeze(1)],
                        dim=1
                    )

                    newly_finished = (
                        next_tokens == dataloader.dataset.vocab.tok2id["<EOS>"]
                    ).squeeze(1)
                    finished = finished | newly_finished

                # Combine the logits for all steps
                logits = torch.stack(all_logits, dim=1)      # shape [batch, seq-1, vocab]
                # If needed, pad them to max_len - 1
                if logits.size(1) < max_len - 1:
                    pad_size = (max_len - 1) - logits.size(1)
                    logits = F.pad(logits, (0, 0, 0, pad_size))
                else:
                    logits = logits[:, : (max_len - 1), :]

                # Flatten
                logits = logits.reshape(-1, logits.size(-1))
                tgt_output = tgt_output.reshape(-1)

                loss = criterion(logits, tgt_output)
                loss.backward()

        # Done teacher forcing or partial decoding
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
) -> Tuple[float, float, float, dict, dict]:
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

    # Initialize dictionaries to store accuracies by length
    length_acc_by_action = {}
    length_acc_by_command = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            action_lengths = batch["action_lengths"]
            command_lengths = batch["command_lengths"]

            tgt_output = tgt[:, 1:]  # Skip BOS token for the output

            # Use greedy decoding
            pred = greedy_decode(
                model, src, eos_idx, bos_idx, pad_idx, device, return_logits=False
            )
            pred = pred[:, 1:]  # Skip BOS token for predictions

            # Calculate accuracies
            for i in range(len(pred)):
                # Token-level and sequence-level accuracies
                token_acc, seq_acc = calculate_accuracy(
                    pred[i].unsqueeze(0), tgt_output[i].unsqueeze(0), pad_idx, eos_idx
                )
                token_accuracies.append(token_acc)
                seq_accuracies.append(seq_acc)

                # Length-specific accuracies
                action_len = action_lengths[i].item()
                command_len = command_lengths[i].item()

                # Action lengths
                if action_len not in length_acc_by_action:
                    length_acc_by_action[action_len] = []
                length_acc_by_action[action_len].append(token_acc)

                # Command lengths
                if command_len not in length_acc_by_command:
                    length_acc_by_command[command_len] = []
                length_acc_by_command[command_len].append(token_acc)

    # Aggregate length-specific accuracies
    greedy_action_acc = {
        k: sum(v) / len(v) for k, v in length_acc_by_action.items()
    }
    greedy_command_acc = {
        k: sum(v) / len(v) for k, v in length_acc_by_command.items()
    }

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc, greedy_action_acc, greedy_command_acc


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

            # Handle both regular Transformer and T5 outputs
            if isinstance(output, torch.Tensor):
                # Regular Transformer output
                output_for_loss = output.contiguous().view(-1, output.shape[-1])
                tgt_output_flat = tgt_output.contiguous().view(-1)
                loss = criterion(output_for_loss, tgt_output_flat)
                pred = output.argmax(dim=-1)
            else:
                # T5 output
                loss = output.loss
                pred = output.logits.argmax(dim=-1)

            total_loss += loss.item()

            # Calculate accuracies
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
) -> Tuple[float, float, float, dict, dict]:
    """
    Evaluates the model using oracle greedy search decoding and groups token-level accuracy by lengths.

    Returns:
        avg_loss: Average loss over the dataset.
        avg_token_acc: Average token-level accuracy.
        avg_seq_acc: Average sequence-level accuracy.
        length_acc_by_action: Token-level accuracy grouped by action sequence length.
        length_acc_by_command: Token-level accuracy grouped by command length.
    """
    model.eval()
    total_loss = 0
    token_accuracies = []
    seq_accuracies = []

    # Initialize dictionaries for length-specific accuracies
    length_acc_by_action = {}
    length_acc_by_command = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Oracle Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            action_lengths = batch["action_lengths"]  # Action sequence lengths
            command_lengths = batch["command_lengths"]  # Command lengths

            tgt_output = tgt[:, 1:]  # Skip BOS token

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
            pred = pred[:, 1:]  # Skip BOS token for predictions

            # Calculate token-level and sequence-level accuracies
            for i in range(len(pred)):
                token_acc, seq_acc = calculate_accuracy(
                    pred[i].unsqueeze(0), tgt_output[i].unsqueeze(0), pad_idx, eos_idx
                )
                token_accuracies.append(token_acc)
                seq_accuracies.append(seq_acc)

                # Group accuracies by lengths
                action_len = action_lengths[i].item()
                command_len = command_lengths[i].item()

                # Action lengths
                if action_len not in length_acc_by_action:
                    length_acc_by_action[action_len] = []
                length_acc_by_action[action_len].append(token_acc)

                # Command lengths
                if command_len not in length_acc_by_command:
                    length_acc_by_command[command_len] = []
                length_acc_by_command[command_len].append(token_acc)

    # Aggregate length-specific accuracies
    length_acc_by_action = {
        k: sum(v) / len(v) for k, v in length_acc_by_action.items()
    }
    length_acc_by_command = {
        k: sum(v) / len(v) for k, v in length_acc_by_command.items()
    }

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_token_acc = sum(token_accuracies) / len(token_accuracies)
    avg_seq_acc = sum(seq_accuracies) / len(seq_accuracies)

    return avg_loss, avg_token_acc, avg_seq_acc, length_acc_by_action, length_acc_by_command


def main(
    train_path: str,
    test_path: str,
    model_suffix: str,
    hyperparams: dict,
    random_seed: int = 42,
    oracle: bool = False,
    train_fn: Callable = train_epoch_teacher_forcing,
    model_class: Type = T5Wrapper,
):
    """
    Main function to train and evaluate the model on the SCAN dataset.
    Now supports both Transformer and T5 models.
    """
    def set_seed(random_seed):
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        np.random.seed(random_seed)

    set_seed(random_seed)

    # Extract DEVICE from hyperparams
    DEVICE = hyperparams["device"]

    # Load datasets and create dataloaders
    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(
        dataset, batch_size=hyperparams["batch_size"], 
        shuffle=True, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"], 
        num_workers=0, pin_memory=True,
    )

    # Dynamic epochs calculation
    data_len = dataset.__len__()
    if (100000 // data_len) > 100:
        hyperparams["epochs"] = (100000 // data_len) 
    else:
        hyperparams["epochs"] = min(20, (100000 // data_len))

    EPOCHS = hyperparams["epochs"]

    # Get special tokens
    src_pad_idx = dataset.src_vocab.tok2id["<PAD>"]
    tgt_pad_idx = dataset.tgt_vocab.tok2id["<PAD>"]
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataset.tgt_vocab.tok2id["<BOS>"]

    # Initialize model using provided model class
    model = model_class(
        src_vocab_size=dataset.src_vocab.vocab_size,
        tgt_vocab_size=dataset.tgt_vocab.vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        max_len=dataset.max_len,
        **hyperparams
    ).to(DEVICE)

    # Initialize optimizer and criterion
    # Only train LoRA adapter parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=hyperparams["learning_rate"])

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
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

    # Load the best model
    checkpoint = torch.load(f"model/best_model_{model_suffix}.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Greedy search evaluation
    _, final_token_acc, final_seq_acc, greedy_action_acc, greedy_command_acc = evaluate_greedy_search(
        model, test_loader, criterion, DEVICE, tgt_eos_idx, tgt_bos_idx, tgt_pad_idx
    )
    print(f"Greedy Accuracy by Action Length: {greedy_action_acc}")
    print(f"Greedy Accuracy by Command Length: {greedy_command_acc}")

    # Initialize oracle_token_acc and oracle_seq_acc to default values
    oracle_token_acc = 0.0
    oracle_seq_acc = 0.0

    # Oracle greedy evaluation (if enabled)
    if oracle:
        _, oracle_token_acc, oracle_seq_acc, oracle_action_acc, oracle_command_acc = evaluate_oracle_greedy_search(
            model, test_loader, criterion, DEVICE, tgt_eos_idx, tgt_bos_idx, tgt_pad_idx
        )
        print(f"Oracle Accuracy by Action Length: {oracle_action_acc}")
        print(f"Oracle Accuracy by Command Length: {oracle_command_acc}")

    return (model, best_accuracy, final_token_acc, final_seq_acc, greedy_action_acc, greedy_command_acc, oracle_token_acc, oracle_seq_acc,)