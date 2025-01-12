from transformers import T5Tokenizer
from torch.utils.data import Dataset

class SCANDataset(Dataset):
    def __init__(self, file_path, tokenizer_name="t5-small", max_len=128):
        self.file_path = file_path
        self.max_len = max_len
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.data = self._load_data()

    def _load_data(self):
        data = []
        with open(self.file_path, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("IN:") and "OUT:" in line:
                    input = line.split("IN:")[1].split("OUT:")[0].strip()
                    output = line.split("OUT:")[1].strip()
                    data.append({"input": input, "output": output})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.tokenizer(
            item["input"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        ).input_ids.squeeze(0)
        labels = self.tokenizer(
            item["output"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        ).input_ids.squeeze(0)
        return {"input_ids": input_ids, "labels": labels}
    
if __name__ == "__main__":
    dataset = SCANDataset("data/simple_split/tasks_train_simple.txt")
    print("Example from dataset:")
    print(f"Input IDs: {dataset[0]['input_ids']}")
    print(f"Labels: {dataset[0]['labels']}")
