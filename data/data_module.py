import lightning as L
from torch.utils.data import DataLoader, random_split
from data.cord import CORDDataset

class CORDDataset(L.LightningDataModule):
    def __init__(self, model, processor, batch_size: int = 1, num_workers: int = 4):
        super().__init__()
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CORDDataset(model=self.model, processor=self.processor, split="train", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                            sort_json_key=False)
            self.valid_dataset = CORDDataset(model=self.model, processor=self.processor, split="validation", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
                            sort_json_key=False)
        # if stage == 'test' or stage is None:
        #     self.test_dataset = CORDDataset(model=self.model, processor=self.processor, split="test", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
        #                     sort_json_key=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # def test_dataloader(self):
    #     return DataLoader()