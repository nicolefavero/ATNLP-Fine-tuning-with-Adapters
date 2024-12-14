import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from dataset import SCANDataset
from model.transformer import Transformer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from torchmetrics import Accuracy
from torchmetrics import Metric

class SequenceAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # Assuming preds and target are both tensors of shape (batch_size, sequence_length)
        # and contain the same type of data (e.g., token indices)
        batch_size = preds.size(0)
        correct_sequences = (preds == target).all(dim=1).sum()
        self.correct += correct_sequences
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total

GRAD_CLIP = 1


def train_epoch_teacher_forcing(model, dataloader, optimizer, criterion, device):
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


def evaluate_greedy_search(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    tgt_eos_idx = dataloader.dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataloader.dataset.tgt_vocab.tok2id["<BOS>"]
    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    accuracy = Accuracy(ignore_index=tgt_pad_idx, task='multiclass', 
                    num_classes=dataloader.dataset.tgt_vocab.vocab_size).to(device)
    seq_accuracy = SequenceAccuracy().to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]

            output = greedy_decode(model, src, tgt_eos_idx, tgt_bos_idx, device, return_logits=True)
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

def evaluate_teacher_forcing(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    accuracy = Accuracy(ignore_index=tgt_pad_idx, task='multiclass', 
                        num_classes=dataloader.dataset.tgt_vocab.vocab_size).to(device)
    seq_accuracy = SequenceAccuracy().to(device)

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

def evaluate_oracle_greedy_search(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    tgt_eos_idx = dataloader.dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataloader.dataset.tgt_vocab.tok2id["<BOS>"]
    tgt_pad_idx = dataloader.dataset.tgt_vocab.tok2id["<PAD>"]
    accuracy = Accuracy(ignore_index=tgt_pad_idx, task='multiclass', 
                    num_classes=dataloader.dataset.tgt_vocab.vocab_size).to(device)
    seq_accuracy = SequenceAccuracy().to(device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Oracle Greedy Search"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_output = tgt[:, 1:]

            output = oracle_greedy_search(model, src, tgt_eos_idx, tgt_bos_idx, tgt_output, device, return_logits=True)
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

@torch.no_grad()
def greedy_decode(model, src, tgt_eos_idx, tgt_bos_idx, device, max_len=128, return_logits=False):
    """
    Perform greedy decoding for a batch of source sequences using the given model.
    
    Args:
        model: The seq2seq transformer model.
        src: Tensor of shape [batch_size, src_seq_len], source sequences.
        tgt_eos_idx: Index of the end-of-sequence token in the target vocabulary.
        tgt_bos_idx: Index of the beginning-of-sequence token in the target vocabulary.
        device: Device to perform computations on.
        max_len: Maximum length of the generated sequences.
        return_logits: Whether to return the logits of each generated token.
    Returns:
        decoded_sequences: A tensor of shape [batch_size, decoded_length] containing the generated token indices.
    """
    model.eval()
    
    batch_size = src.size(0)
    encode_out = model.encoder(src, model.create_src_mask(src))
    pred = torch.full((batch_size, 1), tgt_bos_idx, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_logits = []
    
    for _ in range(max_len - 1):
        if torch.all(finished):
            break
        
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, model.create_src_mask(src), tgt_mask)
        
        last_step_logits = decode_out[:, -1, :]
        all_logits.append(last_step_logits)
        
        next_tokens = torch.argmax(last_step_logits, dim=-1)
        next_tokens = next_tokens.unsqueeze(1)
        pred = torch.cat([pred, next_tokens], dim=1)
        
        newly_finished = next_tokens.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished
    
    if return_logits:
        logits = torch.stack(all_logits, dim=1)
        if logits.size(1) < max_len-1:
            pad_size = (max_len-1) - logits.size(1)
            logits = F.pad(logits, (0, 0, 0, pad_size))
        else:
            logits = logits[:, :(max_len-1), :]
        return logits
    return pred

@torch.no_grad()
def oracle_greedy_search(model, src, tgt_eos_idx, tgt_bos_idx, tgt_output, device, max_len=128, return_logits=False):
    """
    Perform greedy decoding for a batch of source sequences using the given model.
    Uses the target length to forve the model to generate at least the target length.
    """
    model.eval()

    batch_size = src.size(0)
    src_mask = model.create_src_mask(src)
    encode_out = model.encoder(src, src_mask)
    pred = torch.full((batch_size, 1), tgt_bos_idx, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_logits = []
    min_len = _get_min_lengths(tgt_output, tgt_eos_idx)

    for step in range(max_len - 1):
        if torch.all(finished):
            break
        
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, src_mask, tgt_mask)

        logits = decode_out[:, -1, :]
        all_logits.append(logits)

        # Mask EOS tokens for sequences below min length
        current_len = torch.full((batch_size,), step + 1, device=device)
        mask = (current_len < min_len)
        masked_logits = logits.clone()
        masked_logits[mask, tgt_eos_idx] = float('-inf')
        
        next_token = masked_logits.argmax(dim=-1, keepdim=True)
        pred = torch.cat([pred, next_token], dim=1)

        newly_finished = next_token.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished

    if return_logits:
        logits = torch.stack(all_logits, dim=1)
        if logits.size(1) < max_len-1:
            pad_size = (max_len-1) - logits.size(1)
            logits = F.pad(logits, (0, 0, 0, pad_size))
        else:
            logits = logits[:, :(max_len-1), :]
        return logits
    return pred
            


def _get_min_lengths(tgt_output, eos_idx):
    min_lens = []
    for seq in tgt_output:
        eos_positions = torch.nonzero(seq == eos_idx, as_tuple=False)
        if eos_positions.numel() > 0:
            min_len = eos_positions[0].item() + 1
        else:
            # No EOS in the target means we can treat the entire length as min_len
            min_len = seq.size(0)
        min_lens.append(min_len)
    return torch.tensor(min_lens, device=tgt_output.device)


def get_dataset_pairs():
    """Get pairs of training and test dataset paths."""
    base_path = "data/simple_split/size_variations"
    sizes = ["1", "2", "4", "8", "16", "32", "64"]
    pairs = []
    for size in sizes:
        train_path = f"{base_path}/tasks_train_simple_p{size}.txt"
        test_path = f"{base_path}/tasks_test_simple_p{size}.txt"
        pairs.append((train_path, test_path, size))
    return pairs


def main(train_path, test_path, model_suffix, random_seed=42, oracle=False):
    """Modified main function accepting dataset paths and random seed"""
    # Set seeds at the start of main
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # for multi-GPU
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    EMB_DIM = 128
    N_LAYERS = 1
    N_HEADS = 8
    FORWARD_DIM = 512
    DROPOUT = 0.05
    LEARNING_RATE = 7e-4
    BATCH_SIZE = 64
    EPOCHS = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset with provided paths
    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16)

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

    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch_teacher_forcing(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, accuracy, seq_acc = evaluate_teacher_forcing(model, test_loader, criterion, DEVICE)
        g_test_loss, g_accuracy, g_seq_acc = evaluate_greedy_search(model, test_loader, criterion, DEVICE)
        if oracle:
            go_test_loss, go_accuracy, go_seq_acc = evaluate_oracle_greedy_search(model, test_loader, criterion, DEVICE)


        print(f"Dataset p{model_suffix} - Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Greedy Search Loss: {g_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Sequence Length Accuracy: {seq_acc:.4f}")
        print(f"Greedy Search Accuracy: {g_accuracy:.4f}, Sequence Length Accuracy: {g_seq_acc:.4f}")
        if oracle:
            print(f"Oracle Greedy Search Loss: {go_test_loss:.4f}")
            print(f"Oracle Greedy Search Accuracy: {go_accuracy:.4f}, Sequence Length Accuracy: {go_seq_acc:.4f}")
        
        if g_accuracy > best_accuracy:
            best_accuracy = g_accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": g_accuracy,
                },
                f"model/best_model_p{model_suffix}.pt",
            )

        print("-" * 50)

    print(f"Training completed for p{model_suffix}. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy


def run_all_variations():
    """Run training 5 times for all dataset size variations with different seeds"""
    n_runs = 5
    results = {}
    
    # Initialize nested dictionary for results
    for train_path, test_path, size in get_dataset_pairs():
        results[f"p{size}"] = []
    
    # Run training with different seeds
    for run in range(n_runs):
        seed = 42 + run  # Different seed for each run
        print(f"\nStarting run {run+1}/{n_runs} with seed {seed}")
        print("=" * 70)
        
        for train_path, test_path, size in get_dataset_pairs():
            print(f"\nTraining dataset size p{size}")
            accuracy = main(train_path, test_path, size, random_seed=seed)
            results[f"p{size}"].append(accuracy)

    # Print summary with statistics
    print("\nFinal Results Summary:")
    print("=" * 50)
    print("Dataset Size | Mean Accuracy ± Std Dev")
    print("-" * 50)
    
    for size, accuracies in results.items():
        mean = np.mean(accuracies)
        std = np.std(accuracies)
        print(f"{size:11} | {mean:.4f} ± {std:.4f}")
        
        # Print individual runs
        print(f"Individual runs: {', '.join(f'{acc:.4f}' for acc in accuracies)}\n")

if __name__ == "__main__":
    main("data/length_split/tasks_train_length.txt", "data/length_split/tasks_test_length.txt", "length", oracle=True)