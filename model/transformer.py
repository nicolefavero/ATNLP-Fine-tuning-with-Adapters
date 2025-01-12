from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import SCANDataset
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import DistributedSampler

class T5Wrapper(nn.Module):
    def __init__(self, model_name="t5-small", max_len=128):
        super().__init__()
        self.max_len = max_len
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        # Ensure BOS token ID is defined
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token_id = self.tokenizer.pad_token_id
            print(f"Assigned BOS Token ID: {self.tokenizer.bos_token_id}")

    def forward(self, src, tgt=None):
        """
        Forward method for the T5 model.
        - src: Input IDs tensor of shape (batch_size, src_seq_len).
        - tgt: Target IDs tensor of shape (batch_size, tgt_seq_len) for teacher forcing.
        """
        if tgt is not None:
            # Teacher forcing: Provide target sequence for loss calculation
            outputs = self.model(input_ids=src, labels=tgt)
            return outputs.loss, outputs.logits
        else:
            # Inference: Generate sequence without targets
            generated = self.model.generate(input_ids=src, max_length=self.max_len)
            return generated


def main():
    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    # Wrap the model in DDP
    model = T5Wrapper(model_name="t5-small", max_len=128).to(device)
    model = DDP(model, device_ids=[local_rank])

    # Load dataset and dataloader
    train_dataset = SCANDataset("data/simple_split/tasks_train_simple.txt", tokenizer_name="t5-small", max_len=128)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler)

    # Get a batch from the dataset
    batch = next(iter(train_loader))
    src = batch["input_ids"].to(device)
    tgt = batch["labels"].to(device)

    # Forward pass with teacher forcing (training mode)
    with torch.no_grad():
        loss, logits = model(src, tgt)

    # Dynamically determine the vocabulary size from the logits
    logits_vocab_size = logits.size(-1)
    print(f"Logits vocabulary size: {logits_vocab_size}")

    expected_out_shape = torch.Size([src.size(0), tgt.size(1), logits_vocab_size])
    print(f"Expected output shape: {expected_out_shape}")
    print(f"Logits shape: {logits.shape}")

    # Verify output shape
    assert (
        logits.shape == expected_out_shape
    ), f"Wrong output shape, expected: {expected_out_shape}, got: {logits.shape}"

    print("Passed test!")

    # Inference (greedy decoding)
    with torch.no_grad():
        generated_ids = model(src)
        generated_texts = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("\nGenerated Texts:")
    for text in generated_texts:
        print(f"- {text}") 


if __name__ == "__main__":
    main()
