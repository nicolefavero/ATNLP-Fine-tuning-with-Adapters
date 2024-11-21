from torch.utils.data import Dataset 
import torch 

class SCANDataset(Dataset): 
    def __init__(self, file_path, transform=None): 
    self.file_path = file_path 
    self.transform = transform 
    self.data = self._load_data() 
    
    def _load_data(self): 
    data = [] 
    with open(self.file_path, 'r') as file: 
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
        if isinstance(idx, torch.Tensor): 
            idx = idx.item() 
            
            sample = self.data[idx] 
            
            if self.transform: 
                sample = self.transform(sample) 
        
        return sample