import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric


class SequenceAccuracy(Metric):
    def __init__(self, tgt_pad_idx: int = 0, tgt_eos_idx: int = 1) -> None:
        """
        Initializes the SequenceAccuracy metric.

        Args:
            tgt_pad_idx: Index of the padding token in the target vocabulary.
            tgt_eos_idx: Index of the end-of-sequence token in the target vocabulary.
        """
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_eos_idx = tgt_eos_idx

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the metric state with the predictions and targets.

        Args:
            preds: Predicted sequences of shape (batch_size, seq_len).
            target: Target sequences of shape (batch_size, seq_len).
        """
        tgt_eos_idx = self.tgt_eos_idx
        tgt_pad_idx = self.tgt_pad_idx
        batch_size = preds.size(0)
        correct = torch.zeros(batch_size, dtype=torch.bool, device=preds.device)
        for i in range(batch_size):
            pred_seq = preds[i]
            tgt_seq = target[i]
            pred_seq = pred_seq[pred_seq != tgt_pad_idx]
            if tgt_eos_idx in pred_seq:
                pred_seq = pred_seq[
                    : pred_seq.tolist().index(tgt_eos_idx) + 1
                ]  # Remove tokens after <EOS>
            tgt_seq = tgt_seq[tgt_seq != tgt_pad_idx]
            if len(pred_seq) != len(tgt_seq):
                continue
            if torch.all(pred_seq == tgt_seq):
                correct[i] = True
        self.correct += correct.sum()
        self.total += batch_size

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    tgt_eos_idx: int,
    tgt_bos_idx: int,
    device: torch.device,
    max_len: int = 128,
    return_logits: bool = False,
) -> torch.Tensor:
    """
    Performs greedy decoding for a batch of source sequences using the given model.

    Args:
        model: The seq2seq transformer model.
        src: Source sequences tensor of shape (batch_size, src_seq_len).
        tgt_eos_idx: Index of the end-of-sequence token in the target vocabulary.
        tgt_bos_idx: Index of the beginning-of-sequence token in the target vocabulary.
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
    encode_out = model.encoder(src, model.create_src_mask(src))
    pred = torch.full((batch_size, 1), tgt_bos_idx, dtype=torch.long, device=device)

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
        pred = torch.cat([pred, next_tokens], dim=1)

        newly_finished = next_tokens.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished

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
        if torch.all(finished):
            break

        tgt_mask = model.create_tgt_mask(pred)
        decode_out = model.decoder(pred, encode_out, src_mask, tgt_mask)

        logits = decode_out[:, -1, :]
        all_logits.append(logits)

        # Mask EOS tokens for sequences below min length
        current_len = torch.full((batch_size,), step + 1, device=device)
        mask = current_len < min_len
        masked_logits = logits.clone()
        masked_logits[mask, tgt_eos_idx] = float("-inf")

        next_token = masked_logits.argmax(dim=-1, keepdim=True)
        pred = torch.cat([pred, next_token], dim=1)

        newly_finished = next_token.squeeze(1) == tgt_eos_idx
        finished = finished | newly_finished

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
