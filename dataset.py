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
        tokens = list(set(" ".join(self.data).split()))
        id2tok = dict(enumerate(tokens, start=num_special_tokens))
        id2tok.update({v: k for k, v in self.special_tokens.items()})

        return id2tok


class SCANDataset(Dataset):
    def __init__(self, file_path, max_len=128):
        self.file_path = file_path
        self.max_len = max_len
        self.data = self._load_data()
        # Create consolidated vocabulary from both commands and actions
        all_text = [d["command"] for d in self.data] + [d["action"] for d in self.data]
        self.vocab = Vocabulary(all_text)
        self.bos_token = torch.tensor([self.vocab.tok2id["<BOS>"]])
        self.eos_token = torch.tensor([self.vocab.tok2id["<EOS>"]])
        self.pad_token = self.vocab.tok2id["<PAD>"]
        self.unk_token = self.vocab.tok2id["<UNK>"]
        self.encoded_data = [
            {
                "src": self.encode(d["command"], self.vocab),
                "tgt": self.encode(d["action"], self.vocab),
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

    def encode(self, text, vocab):
        tokens = text.split()
        # Vectorized token lookup
        token_ids = [vocab.tok2id.get(tok, self.unk_token) for tok in tokens]
        tokens = torch.tensor(token_ids)

        # Efficient concatenation
        tokens = torch.cat([self.bos_token, tokens, self.eos_token])
        tokens = tokens[: self.max_len - 1]

        # Pad sequence
        padded = torch.nn.functional.pad(
            tokens, (0, self.max_len - len(tokens)), value=self.pad_token
        )
        return padded

    def decode(self, tokens, vocab=None):
        if vocab is None:
            vocab = self.vocab
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
    print(dataset.vocab.vocab_size)  # Now only print once for shared vocab
    print(dataset.decode(dataset[0]["src"], dataset.vocab))
    print(dataset.decode(dataset[0]["tgt"], dataset.vocab))
