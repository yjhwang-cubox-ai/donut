import lightning as L
from torch.utils.data import DataLoader, random_split
from data.cord import CordDataset
from data.business_registration import BRCDataset
from configs.config import config

class DonutDataset(L.LightningDataModule):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self.task_start_token = config.model.task_start_token
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CordDataset(model=self.model,
                                            processor=self.processor,
                                            split="train", 
                                            task_start_token=self.task_start_token, 
                                            sort_json_key=False)
            self.valid_dataset = CordDataset(model=self.model,
                                            processor=self.processor,
                                            split="validation", 
                                            task_start_token=self.task_start_token,
                                            sort_json_key=False)
        # if stage == 'test' or stage is None:
        #     self.test_dataset = CORDDataset(model=self.model, processor=self.processor, split="test", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
        #                     sort_json_key=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=config.data.train_batch_size, shuffle=True, num_workers=config.data.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=config.data.val_batch_size, shuffle=False, num_workers=config.data.num_workers)
    
    # def test_dataloader(self):
    #     return DataLoader()

class DocumentDataset(L.LightningDataModule):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        print(self.processor)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = BRCDataset(
                                    dataset_dir='donut_dataset',
                                    processor=self.processor
                                )
            self.valid_dataset = BRCDataset(
                                    dataset_dir='donut_dataset_val',
                                    processor=self.processor
                                )
        # if stage == 'test' or stage is None:
        #     self.test_dataset = CORDDataset(model=self.model, processor=self.processor, split="test", task_start_token="<s_cord-v2>", prompt_end_token="<s_cord-v2>",
        #                     sort_json_key=False)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=config.data.train_batch_size, shuffle=True, num_workers=config.data.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=config.data.val_batch_size, shuffle=False, num_workers=config.data.num_workers)