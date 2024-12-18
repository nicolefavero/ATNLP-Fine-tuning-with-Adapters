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
        self.src_vocab = Vocabulary([d["command"] for d in self.data])
        self.tgt_vocab = Vocabulary([d["action"] for d in self.data])

    def _load_data(self): 
        data = [] 
        with open(self.file_path, 'r') as file: 
            for line in file: 
                line = line.strip() 
                if line.startswith("IN:") and "OUT:" in line: 
                    input = line.split("IN:")[1].split("OUT:")[0].strip() 
                    output = line.split("OUT:")[1].strip() 
                    data.append({"command": input, "action": output}) 
        return data

    def encode(self, text, vocab):
        tokens = text.split()
        tokens = [vocab.tok2id.get(tok, vocab.tok2id["<UNK>"]) for tok in tokens]
        tokens = [vocab.tok2id["<BOS>"]] + tokens + [vocab.tok2id["<EOS>"]]
        tokens = tokens[:self.max_len-1]
        tokens += [vocab.tok2id["<PAD>"]] * (self.max_len - len(tokens))
        return tokens

    def decode(self, tokens, vocab):
        tokens = [int(tok) for tok in tokens]
        tokens = [vocab.id2tok.get(tok, "<UNK>") for tok in tokens]
        tokens = [tok for tok in tokens if tok not in ["<BOS>", "<EOS>", "<PAD>"]]
        return " ".join(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        src_text = self.data[idx]["command"]
        tgt_text = self.data[idx]["action"]
        src_tokens = self.encode(src_text, self.src_vocab)
        tgt_tokens = self.encode(tgt_text, self.tgt_vocab)
        return {"src": torch.tensor(src_tokens), "tgt": torch.tensor(tgt_tokens)}


if __name__ == "__main__": 
    dataset = SCANDataset("tasks.txt")
    print(dataset[0])
    print(dataset[0]["src"])
    print(dataset[0]["tgt"])
    print(dataset.src_vocab.vocab_size)
    print(dataset.tgt_vocab.vocab_size)
    print(dataset.decode(dataset[0]["src"], dataset.src_vocab))
    print(dataset.decode(dataset[0]["tgt"], dataset.tgt_vocab))
