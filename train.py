import wandb
from dotenv import load_dotenv
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler
from configs.config import config
from models.donut import Donut
from data.data_module import DocumentDataset
from callback.custom_callback import HFTopKModelCheckpoint


def train():
    # 모델 모듈 준비
    model_module = Donut()
    # 데이터셋 모듈 준비
    dataset = DocumentDataset()
    # logger 설정
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)
    # 저장 경로 설정
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

    # trainer 초기화
    trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu",
            devices=8,
            num_nodes=1,
            profiler=profiler,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model_module, datamodule=dataset)
    
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    load_dotenv()    
    train()