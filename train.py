import wandb
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from configs.config import config
from models.donut import Donut
from data.data_module import DonutDataset
from callback.custom_callback import HFTopKModelCheckpoint

def train():
    # 모델 모듈 준비
    model = Donut()
    # 데이터 모듈 준비
    dataset = DonutDataset(model=model.model, processor=model.processor)
    # task_start_token 지정
    tst_token_id = model.processor.tokenizer.convert_tokens_to_ids(config.model.task_start_token)
    model.model.config.decoder_start_token_id = tst_token_id
    # logger 준비
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)
    # 모델 저장 경로
    save_dir = f'logs/{config.wandb.project}/{config.wandb.name}'
    # Callback 준비
    checkpoint_callback = HFTopKModelCheckpoint(
        save_dir=save_dir,
        monitor='val_edit_distance',
        mode='min',
        save_top_k=3,
        training_info=config.wandb.training_info
    )

    trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=dataset)
    
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    load_dotenv()    
    train()