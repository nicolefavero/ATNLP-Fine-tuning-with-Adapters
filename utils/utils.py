import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_accuracy(pred, target, pad_idx, eos_idx):
    """
    Calculate token and sequence accuracy, excluding padding tokens.

    Args:
        pred: Predicted token indices tensor of shape (batch_size, seq_len).
        target: Target token indices tensor of shape (batch_size, seq_len).
        pad_idx: Index of the padding token in the vocabulary.
        eos_idx: Index of the end-of-sequence token in the vocabulary.

    Returns:
        Tuple of token accuracy and sequence accuracy.
    """
    max_len = max(pred.size(1), target.size(1))

    if pred.size(1) < max_len:
        pred = F.pad(pred, (0, max_len - pred.size(1)), value=pad_idx)
    if target.size(1) < max_len:
        target = F.pad(target, (0, max_len - target.size(1)), value=pad_idx)

    # Token accuracy (excluding padding)
    token_mask = target != pad_idx
    correct_tokens = (pred == target) & token_mask
    token_acc = correct_tokens.sum().float() / token_mask.sum().float() if token_mask.sum() > 0 else 0.0

    # Sequence accuracy (up to first EOS, considering padding)
    seq_mask = torch.cumsum(target == eos_idx, dim=1) == 0
    seq_mask = seq_mask & token_mask
    correct_sequences = (pred == target) | ~seq_mask
    seq_acc = torch.all(correct_sequences, dim=1).float().mean()

    return token_acc, seq_acc


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    tgt_eos_idx: int,
    tgt_bos_idx: int,
    tgt_pad_idx: int,
    device: torch.device,
    max_len: int = 128,
    return_logits: bool = False,
) -> torch.Tensor:
    """Performs greedy decoding, now supporting both Transformer and T5."""
    model.eval()

    # Handle T5 model differently
    if isinstance(model, T5Wrapper):
        # Use T5's generate method
        outputs = model.model.generate(
            src,
            max_length=max_len,
            num_beams=1,  # greedy decoding
            pad_token_id=tgt_pad_idx,
            eos_token_id=tgt_eos_idx,
            bos_token_id=tgt_bos_idx,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=return_logits
        )
        if return_logits:
            return outputs.scores
        return outputs.sequences

    # Original Transformer decoding logic
    batch_size = src.size(0)
    encode_out = model.encoder(src, model.create_src_mask(src))
    pred = torch.full((batch_size, 1), tgt_bos_idx, dtype=torch.long, device=device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    all_logits = []

    for _ in range(max_len - 1):
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(
            pred, encode_out, model.create_src_mask(src), tgt_mask
        )

        last_step_logits = decode_out[:, -1, :]
        all_logits.append(last_step_logits)

        next_token = torch.argmax(last_step_logits, dim=-1)

        # Pad finished sequences with EOS index
        next_token = next_token.masked_fill(finished, tgt_pad_idx)
        next_token = next_token.unsqueeze(1)

        pred = torch.cat([pred, next_token], dim=1)

        # Update finished status
        newly_finished = next_token.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished

        if torch.all(finished):
            break

    if return_logits:
        logits = torch.stack(all_logits, dim=1)
        if logits.size(1) < max_len - 1:
            pad_size = (max_len - 1) - logits.size(1)
            logits = F.pad(logits, (0, 0, 0, pad_size))
        else:
            logits = logits[:, : (max_len - 1), :]
        return logits

    return pred


@torch.no_grad()
def oracle_greedy_search(
    model: nn.Module,
    src: torch.Tensor,
    tgt_eos_idx: int,
    tgt_bos_idx: int,
    tgt_pad_idx: int,
    tgt_output: torch.Tensor,
    device: torch.device,
    max_len: int = 128,
    return_logits: bool = False,
) -> torch.Tensor:
    """
    Performs oracle greedy decoding for a batch of source sequences using the given model.
    Ensures the model generates sequences at least as long as the target sequences.

    Args:
        model: The seq2seq transformer model.
        src: Source sequences tensor of shape (batch_size, src_seq_len).
        tgt_eos_idx: Index of the end-of-sequence token in the target vocabulary.
        tgt_bos_idx: Index of the beginning-of-sequence token in the target vocabulary.
        tgt_pad_idx: Index of the padding token in the target vocabulary.
        tgt_output: Target sequences tensor of shape (batch_size, tgt_seq_len).
        device: Device to perform computations on.
        max_len: Maximum length of the generated sequences.
        return_logits: Whether to return the logits of each generated token.

    Returns:
        If return_logits is False:
            Tensor of generated token indices with shape (batch_size, decoded_length).
        If return_logits is True:
            Tensor of logits with shape (batch_size, decoded_length, vocab_size).
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
        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, src_mask, tgt_mask)

        logits = decode_out[:, -1, :]
        all_logits.append(logits)

        # Mask EOS tokens for sequences below min length
        current_len = torch.full((batch_size,), step + 1, device=device)
        mask = current_len < min_len
        masked_logits = logits.clone()
        masked_logits[mask, tgt_eos_idx] = float("-inf")

        # Force EOS if we've reached the target length
        force_eos_mask = (current_len == min_len)
        masked_logits[force_eos_mask, :] = float("-inf")
        masked_logits[force_eos_mask, tgt_eos_idx] = float("inf")

        next_token = torch.argmax(masked_logits, dim=-1)

        # Pad finished sequences with EOS index
        next_token = next_token.masked_fill(finished, tgt_pad_idx)

        next_token = next_token.unsqueeze(1)
        pred = torch.cat([pred, next_token], dim=1)

        # Update finished status
        newly_finished = next_token.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished

        if torch.all(finished):
            break

    if return_logits:
        logits = torch.stack(all_logits, dim=1)
        if logits.size(1) < max_len - 1:
            pad_size = (max_len - 1) - logits.size(1)
            logits = F.pad(logits, (0, 0, 0, pad_size))
        else:
            logits = logits[:, : (max_len - 1), :]
        return logits
    return pred


def _get_min_lengths(tgt_output: torch.Tensor, eos_idx: int) -> torch.Tensor:
    """
    Computes the minimum lengths for each sequence in the batch based on the target sequences.
    """
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
