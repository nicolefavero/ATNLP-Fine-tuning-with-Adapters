from torch.utils.data import Dataset
import torch


class Vocabulary:
    def __init__(self, data, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]):
        self.data = data
        self.special_tokens = {tok: i for i, tok in enumerate(special_tokens)}
        self.id2tok = self._create_id2tok()
        self.tok2id = {v: k for k, v in self.id2tok.items()}
        self.vocab_size = len(self.tok2id)

    def _create_id2tok(self):
        num_special_tokens = len(self.special_tokens)
        tokens = sorted(list(set(" ".join(self.data).split())))
        id2tok = dict(enumerate(tokens, start=num_special_tokens))
        id2tok.update({v: k for k, v in self.special_tokens.items()})

        return id2tok


class SCANDataset(Dataset):
    def __init__(self, file_path, max_len=128):
        self.file_path = file_path
        self.max_len = max_len
        self.data = self._load_data()

        # Create separate vocabularies for commands and actions
        src_text = [d["command"] for d in self.data]
        tgt_text = [d["action"] for d in self.data]
        self.src_vocab = Vocabulary(src_text)
        self.tgt_vocab = Vocabulary(tgt_text)

        # Fetch tokens for both vocabularies
        self.src_bos_token = torch.tensor([self.src_vocab.tok2id["<BOS>"]])
        self.src_eos_token = torch.tensor([self.src_vocab.tok2id["<EOS>"]])
        self.src_pad_token = self.src_vocab.tok2id["<PAD>"]
        self.src_unk_token = self.src_vocab.tok2id["<UNK>"]

        self.tgt_bos_token = torch.tensor([self.tgt_vocab.tok2id["<BOS>"]])
        self.tgt_eos_token = torch.tensor([self.tgt_vocab.tok2id["<EOS>"]])
        self.tgt_pad_token = self.tgt_vocab.tok2id["<PAD>"]
        self.tgt_unk_token = self.tgt_vocab.tok2id["<UNK>"]

        # Encode data
        self.encoded_data = [
            {
                "src": self.encode(d["command"], self.src_vocab, is_src=True),
                "tgt": self.encode(d["action"], self.tgt_vocab, is_src=False),
            }
            for d in self.data
        ]

    def _load_data(self):
        data = []
        with open(self.file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("IN:") and "OUT:" in line:
                    input = line.split("IN:")[1].split("OUT:")[0].strip()
                    output = line.split("OUT:")[1].strip()
                    data.append({"command": input, "action": output})
        return data

    def encode(self, text, vocab, is_src=True):
        tokens = text.split()
        # Vectorized token lookup
        unk_token = self.src_unk_token if is_src else self.tgt_unk_token
        token_ids = [vocab.tok2id.get(tok, unk_token) for tok in tokens]
        tokens = torch.tensor(token_ids)

        # Add BOS/EOS tokens
        bos_token = self.src_bos_token if is_src else self.tgt_bos_token
        eos_token = self.src_eos_token if is_src else self.tgt_eos_token
        tokens = torch.cat([bos_token, tokens, eos_token])
        tokens = tokens[: self.max_len - 1]

        # Pad sequence
        pad_token = self.src_pad_token if is_src else self.tgt_pad_token
        padded = torch.nn.functional.pad(
            tokens, (0, self.max_len - len(tokens)), value=pad_token
        )
        return padded

    def decode(self, tokens, is_src=True):
        vocab = self.src_vocab if is_src else self.tgt_vocab
        tokens = [int(tok) for tok in tokens]
        tokens = [vocab.id2tok.get(tok, "<UNK>") for tok in tokens]
        tokens = [tok for tok in tokens if tok not in ["<BOS>", "<EOS>", "<PAD>"]]
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return {
                "src": torch.stack(
                    [
                        self.encoded_data[i]["src"].clone()
                        for i in range(*idx.indices(len(self)))
                    ]
                ),
                "tgt": torch.stack(
                    [
                        self.encoded_data[i]["tgt"].clone()
                        for i in range(*idx.indices(len(self)))
                    ]
                ),
            }
        return {
            "src": self.encoded_data[idx]["src"].clone(),
            "tgt": self.encoded_data[idx]["tgt"].clone(),
        }


if __name__ == "__main__":
    dataset = SCANDataset("data/simple_split/tasks_train_simple.txt")
    print(dataset[0])
    print(dataset[0]["src"])
    print(dataset[0]["tgt"])
    print(f"Source vocabulary size: {dataset.src_vocab.vocab_size}")
    print(f"Target vocabulary size: {dataset.tgt_vocab.vocab_size}")
    print(dataset.decode(dataset[0]["src"], is_src=True))
    print(dataset.decode(dataset[0]["tgt"], is_src=False))
    # Print both vocabularies sorted by id
    print(
        "Source vocabulary:",
        sorted(dataset.src_vocab.id2tok.items(), key=lambda x: x[0]),
    )
    print(
        "Target vocabulary:",
        sorted(dataset.tgt_vocab.id2tok.items(), key=lambda x: x[0]),
    )
