import os
import json
import shutil
import lightning as L
from lightning.fabric.utilities.rank_zero import rank_zero_only

class HFTopKModelCheckpoint(L.Callback):
    def __init__(self, 
                save_dir: str,
                monitor: str = "val_loss",
                mode: str = "min",
                save_top_k: int = 3,
                training_info=None):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.top_k = save_top_k
        self.training_info = training_info
        
        self.top_k_models = []
        
        # 메인 프로세스에서만 기존 체크포인트를 삭제하고 training info를 저장
        self._cleanup_save_dir()
        # 모든 프로세스에서 save_dir이 존재하도록 함
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 메인 프로세스가 아니라면 실행하지 않음
        if not trainer.is_global_zero:
            return
        # trainer.callback_metrics에서 validation metric 가져오기
        logs = trainer.callback_metrics
        current_score = logs.get(self.monitor)
        if current_score is None:
            return

        if hasattr(current_score, "item"):
            current_score = current_score.item()

        qualifies = False
        # top_k 모델 수가 아직 미달이면 무조건 저장 대상에 포함
        if len(self.top_k_models) < self.top_k:
            qualifies = True
        else:
            # 이미 top_k개가 저장되어 있으면, 현재 스코어가 기존 중 가장 '나쁜' 스코어보다 개선되었는지 판단
            if self.mode == "min":
                worst_score = max(score for score, _ in self.top_k_models)
                if current_score < worst_score:
                    qualifies = True
            else:
                worst_score = min(score for score, _ in self.top_k_models)
                if current_score > worst_score:
                    qualifies = True

        if qualifies:
            epoch = trainer.current_epoch
            step = trainer.global_step
            # 디렉토리 이름 생성 (epoch, step, 점수를 포함)
            model_dir = os.path.join(self.save_dir, f"epoch-{epoch:03d}_step-{step:05d}_ED-{current_score:.4f}")
            os.makedirs(model_dir, exist_ok=True)
            if self.top_k_models:
                print(
                    f"\nValidation '{self.monitor}' improved to {current_score:.4f}. "
                    f"Checkpoint saved at {model_dir}. (Current {self.monitor}: {current_score:.4f}, "
                    f"Best {self.monitor} so far: {self.top_k_models[0][0]:.4f})"
                )
            else:
                print(f"\nValidation '{self.monitor}' improved to {current_score:.4f}. Checkpoint saved at {model_dir}.")
                
            # Hugging Face 방식으로 모델과 토크나이저 저장
            pl_module.model.save_pretrained(model_dir)
            pl_module.processor.save_pretrained(model_dir)

            # 현재 checkpoint 정보를 리스트에 추가
            self.top_k_models.append((current_score, model_dir))
            # 성능 기준으로 정렬 (mode에 따라 정렬 순서 달라짐)
            if self.mode == "min":
                self.top_k_models.sort(key=lambda x: x[0])  # 낮은 loss가 앞쪽에 위치
            else:
                self.top_k_models.sort(key=lambda x: -x[0])  # 높은 score가 앞쪽에 위치

            # top_k 개수를 초과하는 경우, 가장 '나쁜' checkpoint 삭제
            if len(self.top_k_models) > self.top_k:
                worst = self.top_k_models.pop()  # 리스트의 마지막 요소가 가장 낮은 성능
                shutil.rmtree(worst[1])
                print(f"Checkpoint {worst[1]} removed as it exceeded the top {self.top_k} limit.")
    
    @rank_zero_only
    def _cleanup_save_dir(self):
        if os.path.exists(self.save_dir):
            for name in os.listdir(self.save_dir):
                path = os.path.join(self.save_dir, name)
                if os.path.isdir(path) and name.startswith("epoch-"):
                    shutil.rmtree(path)
        else:
            os.makedirs(self.save_dir)
        self._save_training_info()
    
    def _save_training_info(self):
        info_to_save = (
            vars(self.training_info)
            if hasattr(self.training_info, '__dict__')
            else self.training_info
        )
        info_path = os.path.join(self.save_dir, "training_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info_to_save, f, indent=4, ensure_ascii=False)
        print(f"Saved training info at {info_path}")