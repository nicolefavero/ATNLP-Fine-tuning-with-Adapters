import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader, random_split
from dataset import SCANDataset
from model.transformer import Transformer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

GRAD_CLIP = 1

def train_epoch(model, dataloader, optimizer, criterion, device):
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

def evaluate(model, dataloader, criterion, device):
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

def calculate_accuracy(model, test_loader, dataset, device):
    tgt_eos_idx = dataset.tgt_vocab.tok2id["<EOS>"]
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_targets = []
        for batch in test_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            output = output.argmax(dim=-1)
            all_preds.extend(output.cpu().numpy().tolist())
            all_targets.extend(tgt_output.cpu().numpy().tolist())

    # Filter and process predictions
    filtered_preds = []
    filtered_targets = []
    for i in range(len(all_preds)):
        pred_seq = []
        for j in range(len(all_preds[i])):
            if all_preds[i][j] == tgt_eos_idx:
                break
            pred_seq.append(all_preds[i][j])
        filtered_preds.append(pred_seq)
        
        target_seq = []
        for j in range(len(pred_seq)):
            target_seq.append(all_targets[i][j])
        filtered_targets.append(target_seq)

    flat_preds = [item for sublist in filtered_preds for item in sublist]
    flat_targets = [item for sublist in filtered_targets for item in sublist]
    
    return accuracy_score(flat_targets, flat_preds)

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

def main(train_path, test_path, model_suffix):
    """Modified main function accepting dataset paths"""
    EMB_DIM = 128
    N_LAYERS = 1
    N_HEADS = 8
    FORWARD_DIM = 512
    DROPOUT = 0.05
    LEARNING_RATE = 7e-4
    BATCH_SIZE = 64
    EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset with provided paths
    dataset = SCANDataset(train_path)
    test_dataset = SCANDataset(test_path)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
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
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_vocab.tok2id["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss = evaluate(model, test_loader, criterion, DEVICE)
        accuracy = calculate_accuracy(model, test_loader, dataset, DEVICE)
        
        print(f"Dataset p{model_suffix} - Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print(f"New best accuracy: {best_accuracy:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, f'best_model_p{model_suffix}.pt')
        
        print("-" * 50)

    print(f"Training completed for p{model_suffix}. Best accuracy: {best_accuracy:.4f}")
    return best_accuracy

def run_all_variations():
    """Run training for all dataset size variations"""
    results = {}
    for train_path, test_path, size in get_dataset_pairs():
        print(f"\nStarting training for dataset size p{size}")
        print("=" * 70)
        accuracy = main(train_path, test_path, size)
        results[f"p{size}"] = accuracy
    
    # Print summary of results
    print("\nFinal Results Summary:")
    print("=" * 30)
    for size, acc in results.items():
        print(f"Dataset {size}: {acc:.4f}")

if __name__ == "__main__":
    main("data/simple_split/tasks_train_simple.txt", "data/simple_split/tasks_test_simple.txt",100)
