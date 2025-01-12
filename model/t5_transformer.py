import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

class T5Wrapper(nn.Module):
    """Wrapper around HuggingFace's T5 model to maintain compatibility with existing code."""
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        model_name="t5-small",
        max_len=128,
        model_config=None,  # Add this parameter
        **kwargs
    ):
        super().__init__()
        
        # Use custom config if provided, otherwise use default from model_name
        if model_config is not None:
            self.model = T5ForConditionalGeneration(model_config)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_len = max_len

        # Resize token embeddings if needed
        self.model.resize_token_embeddings(max(src_vocab_size, self.tokenizer.vocab_size))


    def forward(self, src, tgt):
        # Convert padding tokens to T5's padding token
        src = torch.where(src == self.src_pad_idx, self.tokenizer.pad_token_id, src)
        tgt = torch.where(tgt == self.tgt_pad_idx, self.tokenizer.pad_token_id, tgt)

        # Create attention masks
        src_mask = (src != self.tokenizer.pad_token_id).long()
        tgt_mask = (tgt != self.tokenizer.pad_token_id).long()

        # Forward pass through T5
        outputs = self.model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=tgt,
            decoder_attention_mask=tgt_mask,
            labels=tgt,
            return_dict=True
        )

        return outputs.logits

    def create_src_mask(self, src):
        # T5 handles masking internally, but we keep this for compatibility
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt):
        # T5 handles masking internally, but we keep this for compatibility
        return (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

    def encoder(self, src, src_mask):
        # For compatibility with the original architecture
        # In practice, T5 handles encoding internally
        pass

    def decoder(self, tgt, encoder_out, src_mask, tgt_mask):
        # For compatibility with the original architecture
        # In practice, T5 handles decoding internally
        pass