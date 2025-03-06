import wandb
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler
from configs.config import config
from models.donut import Donut
from data.data_module import DonutDataset, DocumentDataset
from callback.custom_callback import HFTopKModelCheckpoint


def train():
    # 모델 모듈 준비
    model = Donut()
    # 데이터 모듈 준비
    dataset = DocumentDataset(model=model.model, processor=model.processor)
    # logger 준비
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)
    # 모델 저장 경로
    save_dir = f'logs/{config.wandb.project}/{config.wandb.name}'
    # Callback 준비
    checkpoint_callback = HFTopKModelCheckpoint(
        save_dir=save_dir,
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        training_info=config.wandb.training_info
    )

    profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            profiler=profiler,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=dataset)
    
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    load_dotenv()    
    train()