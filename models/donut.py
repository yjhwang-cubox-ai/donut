import re
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from nltk import edit_distance
from models.model_setup import init_model_and_processor
from configs.config import config
import time

class Donut(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
    
    def setup(self, stage=None):
        self.model, self.processor = init_model_and_processor()
    
    def training_step(self, batch, batch_idx):
        print(f"[시간 측정] training_step 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        pixel_values, labels = batch['pixel_values'], batch['labels']
        
        outputs = self.model(pixel_values, labels=labels)
        train_loss = outputs.loss
        self.log("train_loss", train_loss)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch['pixel_values'], batch['labels']
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)

        outputs = self.model(pixel_values, labels=labels)
        val_loss = outputs.loss
        self.log("val_loss", val_loss)
        return val_loss
        
        # outputs = self.model.generate(pixel_values,
        #                             decoder_input_ids=decoder_input_ids,
        #                             max_length=config.model.max_length,                                    
        #                             num_beams=1,
        #                             early_stopping=False,
        #                             pad_token_id=self.processor.tokenizer.pad_token_id,
        #                             eos_token_id=self.processor.tokenizer.eos_token_id,
        #                             use_cache=True,
        #                             bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
        #                             return_dict_in_generate=True,)
    
        # predictions = []
        # for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
        #     seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        #     seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        #     predictions.append(seq)

        # gt = []
        # for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
        #     seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        #     seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
        #     gt.append(seq)

        # scores = []
        # for pred, gt in zip(predictions, gt):
        #     pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
        #     # NOT NEEDED ANYMORE
        #     # answer = re.sub(r"<.*?>", "", answer, count=1)
        #     # answer = answer.replace(self.processor.tokenizer.eos_token, "")
        #     scores.append(edit_distance(pred, gt) / max(len(pred), len(gt)))

        #     if config.training.verbose==False and len(scores) == 1:
        #         print(f"\nPrediction: {pred}")
        #         print(f"\n    GT: {gt}")
        #         print(f"\n Normed ED: {scores[0]}\n")

        # self.log("val_edit_distance", np.mean(scores))
        
        # return scores 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.training.lr)
        return optimizer