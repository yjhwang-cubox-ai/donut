import wandb
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from configs.config import config
from models.donut import Donut
from data.data_module import DonutDataset

def train():
    # 모델 모듈 준비
    model = Donut()
    # 데이터 모듈 준비
    dataset = DonutDataset(model=model.model, processor=model.processor)
    # logger 준비
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)
    # Callback 준비
    callbacks = [
        ModelCheckpoint(
            monitor='val_edit_distance',
            mode='min',
            save_top_k=3,
            filename='{epoch}-{step}-{val_edit_distance:.3f}',
            verbose=True
        )
    ]

    trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            logger=wandb_logger,
            callbacks=callbacks
    )

    trainer.fit(model=model, datamodule=dataset)
    
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    load_dotenv()    
    train()