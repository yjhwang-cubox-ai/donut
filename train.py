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
import time
from models.model_setup import init_model_and_processor


def train():
    # 전체 시작 시간 측정
    total_start_time = time.time()
    print(f"[시간 측정] 학습 시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 모델 모듈 준비
    model_start_time = time.time()
    model_module = Donut()
    model_end_time = time.time()
    print(f"[시간 측정] 모델 초기화 시간: {model_end_time - model_start_time:.2f}초")

    # 데이터 모듈 준비
    data_start_time = time.time()
    dataset = DocumentDataset()
    data_end_time = time.time()
    print(f"[시간 측정] 데이터 모듈 초기화 시간: {data_end_time - data_start_time:.2f}초")

    # logger 준비
    logger_start_time = time.time()
    wandb_logger = WandbLogger(project=config.wandb.project, name=config.wandb.name)
    logger_end_time = time.time()
    print(f"[시간 측정] 로거 초기화 시간: {logger_end_time - logger_start_time:.2f}초")
    # 모델 저장 경로
    save_dir = f'logs/{config.wandb.project}/{config.wandb.name}'
    # Callback 준비
    callback_start_time = time.time()
    checkpoint_callback = HFTopKModelCheckpoint(
        save_dir=save_dir,
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        training_info=config.wandb.training_info
    )
    callback_end_time = time.time()
    print(f"[시간 측정] 콜백 초기화 시간: {callback_end_time - callback_start_time:.2f}초")

    profiler = SimpleProfiler(dirpath=".", filename="perf_logs")

    # 트레이너 초기화
    trainer_start_time = time.time()
    trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator="gpu",
            devices=8,
            num_nodes=1,
            profiler=profiler,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
    )
    trainer_end_time = time.time()
    print(f"[시간 측정] 트레이너 초기화 시간: {trainer_end_time - trainer_start_time:.2f}초")

    # 실제 학습 시작 직전 시간
    before_fit_time = time.time()
    print(f"[시간 측정] 학습 직전까지 총 준비 시간: {before_fit_time - total_start_time:.2f}초")

    trainer.fit(model=model_module, datamodule=dataset)
    
    wandb.finish()

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    load_dotenv()    
    train()