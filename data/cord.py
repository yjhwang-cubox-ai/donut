import json
import random
from typing import Any, List, Tuple
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from models.model_setup import init_processor
from configs.config import config

class CordDataset(Dataset):
    def __init__(self,
                split: str = "train",
                ignore_id: int = -100,
                task_start_token:str = "<s>",
                prompt_end_token:str = None,
                sort_json_key: bool = True,
                processor = None,
                model = None
    ):
        super().__init__()
        
        self.model = model
        self.processor = processor
        self.max_length = config.model.max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token  else task_start_token
        self.sort_json_key = sort_json_key
        
        # self.dataset = load_dataset(dataset_name_or_path, split=self.split, cache_dir="./hub/dataset")
        self.dataset = load_dataset(config.data.dataset_name_or_path, split=self.split, cache_dir=config.data.cache_dir)
        self.dataset_length = len(self.dataset)
        
        self.gt_token_sequences = []
        for sample in self.dataset:
            ground_truth = json.loads(sample['ground_truth'])
            if "gt_parses" in ground_truth:
                assert isinstance(ground_truth['gt_parses'], list)
                gt_jsons = ground_truth['gt_parses']
            else:
                assert "gt_parse" in ground_truth and isinstance(ground_truth['gt_parse'], dict)
                gt_jsons = [ground_truth['gt_parse']]
            
            self.gt_token_sequences.append(
                [
                    self.json2token(
                        gt_json, 
                        update_special_tokens_for_json_key=self.split == "train",
                        sort_json_key=self.sort_json_key,
                    ) 
                    + self.processor.tokenizer.eos_token
                    for gt_json in gt_jsons 
                ]
            )
            
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.processor.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
    
    def json2token(self, obj:Any, update_special_tokens_for_json_key:bool = True, sort_json_key:bool = True):
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"<s_{k}>", fr"</s_{k}>"])
                    output += (
                        fr"<s_{k}>"
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            # 만약 이미 특별 토큰 리스트에 있다면 해당 형식으로 변환
            if f"<{obj}/>" in self.processor.tokenizer.get_added_vocab():
                obj = f"<{obj}/>"  # 범주형 특별 토큰일 경우
            return obj
    
    def add_tokens(self, list_of_tokens:List[str]):
        newly_added_num = self.processor.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.processor.tokenizer))
    
    def __len__(self) -> int:
        return self.dataset_length
    
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.dataset[index]
        
        # input
        pixel_values = self.processor(sample["image"], return_tensors="pt", legacy=False, add_special_tokens=True).pixel_values
        pixel_values = pixel_values.squeeze()
        
        # target
        target_sequence = random.choice(self.gt_token_sequences[index])
        input_ids = self.processor.tokenizer(
            target_sequence,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.ignore_id # model does not need to predict pad token
        
        return pixel_values, labels, target_sequence