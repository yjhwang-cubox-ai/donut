import os
import operator
import shutil
import json
import pytorch_lightning as pl

class SaveHFModelOnMetricCallback(pl.Callback):
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_edit_distance',
        mode: str = 'min',
        save_top_k: int = 3,
        filename: str = '{epoch}-{step}-{val_edit_distance:.3f}',
        training_info=None  # namespace 또는 dict 형식으로 입력됨
    ):
        """
        Args:
            save_dir: 체크포인트를 저장할 기본 디렉토리.
            monitor: 모니터링할 metric 이름.
            mode: 'min' 또는 'max'. metric이 낮을수록 좋은지, 높은게 좋은지.
            save_top_k: 보관할 체크포인트 수.
            filename: 체크포인트 디렉토리 이름에 사용할 포맷. 
                    예시: '{epoch}-{step}-{val_edit_distance:.3f}'
            training_info: 모델 학습에 사용한 파라미터, 목적 등 추가 정보를 담은 namespace 또는 dict.
        """
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.training_info = training_info

        # 저장된 체크포인트 목록: (metric_value, checkpoint_path) 튜플 리스트
        self.top_k = []
        # 비교 연산자: mode가 'min'이면 operator.lt, 'max'이면 operator.gt 사용.
        self.compare_op = operator.lt if mode == 'min' else operator.gt

        os.makedirs(self.save_dir, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # trainer.callback_metrics 에 모니터링할 metric이 기록되어 있어야 합니다.
        metric_value = trainer.callback_metrics.get(self.monitor)
        if metric_value is None:
            return

        # 현재 epoch과 global step을 가져옵니다.
        epoch = trainer.current_epoch
        step = trainer.global_step

        # filename 포맷에 맞게 이름 생성 (키워드 인자로 metric 값을 전달)
        try:
            checkpoint_name = self.filename.format(epoch=epoch, step=step, **{self.monitor: metric_value})
        except KeyError:
            checkpoint_name = f'{epoch}-{step}-{metric_value:.3f}'

        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)

        # top_k 갯수가 부족하면 무조건 저장.
        if len(self.top_k) < self.save_top_k:
            self._save_checkpoint(pl_module, checkpoint_path)
            self.top_k.append((metric_value, checkpoint_path))
        else:
            # top_k에 이미 save_top_k 개의 체크포인트가 있을 경우, 최악의 체크포인트와 비교
            if self.mode == 'min':
                worst = max(self.top_k, key=lambda x: x[0])
            else:
                worst = min(self.top_k, key=lambda x: x[0])
            
            # 새 metric이 기존 최악의 값보다 좋으면 교체
            if self.compare_op(metric_value, worst[0]):
                self._remove_checkpoint(worst[1])
                self.top_k.remove(worst)
                self._save_checkpoint(pl_module, checkpoint_path)
                self.top_k.append((metric_value, checkpoint_path))

    def _save_checkpoint(self, pl_module, checkpoint_path: str):
        """Hugging Face의 save_pretrained 메서드를 사용해 모델, 토크나이저, 그리고 학습 메타 정보를 저장합니다."""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # 토크나이저의 토큰 수에 맞춰 임베딩 크기를 재조정
        new_token_count = len(pl_module.processor.tokenizer)
        pl_module.model.decoder.resize_token_embeddings(new_token_count)
        
        # 모델과 토크나이저 저장
        pl_module.model.save_pretrained(checkpoint_path)
        pl_module.processor.tokenizer.save_pretrained(checkpoint_path)
        
        # training_info 정보가 있다면, JSON 파일로 저장합니다.
        if self.training_info:
            # namespace 객체라면 dict로 변환합니다.
            info_to_save = vars(self.training_info) if hasattr(self.training_info, '__dict__') else self.training_info
            info_path = os.path.join(checkpoint_path, "training_info.json")
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info_to_save, f, indent=4, ensure_ascii=False)
            print(f"Saved training info at {info_path}")
        
        print(f"Saved checkpoint at {checkpoint_path}")

    def _remove_checkpoint(self, checkpoint_path: str):
        """저장된 체크포인트 디렉토리를 삭제합니다."""
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
            print(f"Removed checkpoint at {checkpoint_path}")
