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


def evaluate_teacher_forcing(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_epoch_mix(model, dataloader, optimizer, criterion, device, p_greedy=0.3):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        optimizer.zero_grad()
                
        if np.random.rand() < p_greedy:
            encoder_out = model.encoder(src, model.create_src_mask(src))
            decoder_input = tgt_input[:, 0].unsqueeze(1)
            outputs = []

            for i in range(tgt_input.size(1)):
                tgt_mask = model.create_tgt_mask(decoder_input)
                step_output = model.decoder(decoder_input, encoder_out, 
                                         model.create_src_mask(src), tgt_mask)
                outputs.append(step_output[:, -1:, :])
                
                if i < tgt_input.size(1) - 1:
                    next_token = step_output[:, -1:].argmax(-1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
            output = torch.cat(outputs, dim=1)
        else:
            output = model(src, tgt_input)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(output, tgt_output)
        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def greedy_decode(model, src, tgt_all, dataset, device, max_len=128):
    """
    Perform greedy decoding for a batch of source sequences using the given model.
    
    Args:
        model: The seq2seq transformer model.
        src: Tensor of shape [batch_size, src_seq_len], source sequences.
        dataset: The dataset object that contains vocab information.
        device: Torch device (cuda or cpu).
        max_len: Maximum decoding length.

    Returns:
        decoded_sequences: A tensor of shape [batch_size, decoded_length] containing the generated token indices.
    """
    model.eval()
    
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_bos_idx = dataset.tgt_vocab.tok2id["<BOS>"]
    
    batch_size = src.size(0)


    encode_out = model.encoder(src, model.create_src_mask(src))
    pred = torch.full((batch_size, 1), tgt_bos_idx, dtype=torch.long, device=device)
    
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, model.create_src_mask(src), tgt_mask)
        
        last_step_logits = decode_out[:, -1, :]

        next_tokens = torch.argmax(last_step_logits, dim=-1)
        
        next_tokens = next_tokens.unsqueeze(1)
        pred = torch.cat([pred, next_tokens], dim=1) 
        
        newly_finished = next_tokens.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished
        
        if torch.all(finished):
            break
    
    return pred

def constrained_greedy_search(model, test_loader, dataset, device):
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    tgt_pad_idx = dataset.tgt_vocab.tok2id["<PAD>"]
    tgt_bos_idx = dataset.tgt_vocab.tok2id["<BOS>"]
    max_len = dataset.max_len

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            src = batch["src"].to(device) 
            tgt = batch["tgt"].to(device)   

            tgt_output = tgt[:, 1:] 

            min_lens = _get_min_lengths(tgt_output, tgt_eos_idx)

            src_mask = model.create_src_mask(src)
            encode_out = model.encoder(src, src_mask)

            batch_size = src.size(0)
            generated = torch.full((batch_size, 1),
                                   tgt_bos_idx,
                                   device=device,
                                   dtype=torch.long)

            for step in range(max_len):
                tgt_mask = model.create_tgt_mask(generated) 
                decode_out = model.decoder(generated, encode_out, src_mask, tgt_mask)

                logits = decode_out[:, -1, :]

                mask = (step < min_lens)  
                logits[mask, tgt_eos_idx] = float('-inf')

                next_token = logits.argmax(dim=-1, keepdim=True) 
                generated = torch.cat([generated, next_token], dim=1)

            all_preds.extend(generated.cpu().numpy().tolist())
            all_targets.extend(tgt_output.cpu().numpy().tolist())

    return process_predictions(all_preds, all_targets, tgt_eos_idx, tgt_pad_idx)

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

def process_predictions(all_preds, all_targets, tgt_eos_idx, tgt_pad_idx):
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().tolist()
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().tolist()

    flat_preds = []
    flat_targets = []
    length_matches = []

    for i in range(len(all_preds)):
        pred_seq = all_preds[i]
        target_seq = all_targets[i]

        if tgt_eos_idx in pred_seq:
            pred_eos_idx = pred_seq.index(tgt_eos_idx) + 1
        else:
            pred_eos_idx = len(pred_seq)
        
        if tgt_eos_idx in target_seq:
            target_eos_idx = target_seq.index(tgt_eos_idx) + 1
        else:
            target_eos_idx = len(target_seq)

        pred_seq = pred_seq[:pred_eos_idx]
        target_seq = target_seq[:target_eos_idx]

        length_matches.append(int(len(pred_seq) == len(target_seq)))

        max_len = max(len(pred_seq), len(target_seq))
        pred_seq.extend([tgt_pad_idx] * (max_len - len(pred_seq)))
        target_seq.extend([tgt_pad_idx] * (max_len - len(target_seq)))

        flat_preds.extend(pred_seq)
        flat_targets.extend(target_seq)
    
    seq_len_acc = sum(length_matches) / len(length_matches) if length_matches else 0.0

    token_accuracy = accuracy_score(flat_targets, flat_preds)
    return token_accuracy, seq_len_acc


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


def main(train_path, test_path, model_suffix, random_seed=42):
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
        train_loss = train_epoch_mix(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate_teacher_forcing(model, test_loader, criterion, DEVICE)
        for batch in test_loader:
            src = batch["src"].to(DEVICE)
            tgt = batch["tgt"].to(DEVICE)
            break
        pred = greedy_decode(model, src, tgt, test_dataset, DEVICE)
        accuracy, seq_acc = process_predictions(pred, tgt[:, 1:], test_dataset.tgt_vocab.tok2id["<EOS>"], test_dataset.tgt_vocab.tok2id["<PAD>"])

        # fixed_accuracy, fixed_seq_acc = constrained_greedy_search(
        #     model, test_loader, test_dataset, DEVICE
        # )

        print(f"Dataset p{model_suffix} - Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}\nSequence Length Accuracy: {seq_acc:.4f}")
        # print(f"Fixed Length Accuracy: {fixed_accuracy:.4f}\nSequence Length Accuracy: {fixed_seq_acc:.4f}")

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
    run_all_variations()