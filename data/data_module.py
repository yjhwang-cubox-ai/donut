import lightning as L
from torch.utils.data import DataLoader, random_split

class CORDDataset(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        
    def prepare_data(self):
        pass
    
    def setup(self, stage):
        pass
    
    def train_dataloader(self):
        return DataLoader()
    
    def val_dataloader(self):
        return DataLoader()
    
    def test_dataloader(self):
        return DataLoader()