import re
import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from nltk import edit_distance
from models.model_setup import init_model_config, init_processor, init_model
from configs.config import config

class Donut(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model_config = init_model_config()
        self.processor = init_processor()
        self.model = init_model(self.model_config, self.processor)
    
    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch['pixel_values'], batch['labels']
        
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]
        # we feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        
        outputs = self.model.generate(pixel_values,
                                    decoder_input_ids=decoder_input_ids,
                                    max_length=config.model.max_length,                                    
                                    num_beams=1,
                                    early_stopping=False,
                                    pad_token_id=self.processor.tokenizer.pad_token_id,
                                    eos_token_id=self.processor.tokenizer.eos_token_id,
                                    use_cache=True,
                                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                    return_dict_in_generate=True,)
    
        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            # NOT NEEDED ANYMORE
            # answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if config.training.verbose==False and len(scores) == 1:
                print(f"\nPrediction: {pred}")
                print(f"\n    Answer: {answer}")
                print(f"\n Normed ED: {scores[0]}\n")

        self.log("val_edit_distance", np.mean(scores))
        
        return scores 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.training.lr)
        return optimizer
    
    # def on_train_start(self):
    #     print("학습을 시작합니다!\n")
    #     print("pad_token_id 와 decoder_start_token_id 를 설정합니다!\n")
    #     self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
    #     self.model.config.decoder_start_token_id = self.processor.tokenizer.convert_tokens_to_ids(['<s_cord-v2>'])[0]