import os
import json
from PIL import Image
from torch.utils.data import Dataset
from configs.config import config

class BRCDataset(Dataset):
    def __init__(self, dataset_dir, processor):
        """
        dataset_dir: donut_dataset 폴더 경로
        processor: DonutProcessor
        """
        self.processor = processor
        self.max_length = config.model.max_length

        annotation_dir = os.path.join(dataset_dir, "donut_format")
        annotation_files = [
            f for f in os.listdir(annotation_dir) if f.endswith(".json")
        ]

        self.annotations = []
        for ann_file in annotation_files:
            ann_path = os.path.join(annotation_dir, ann_file)
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.annotations.append(data)
        
        # 이미지가 저장된 디렉토리 경로
        self.images_dir = os.path.join(dataset_dir, "images")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        ann_image_path = ann["image"]
        image_path = os.path.join(self.images_dir, os.path.basename(ann_image_path))
        ground_truth = ann["ground_truth"]

        # 실제 이미지 경로 구성
        image = Image.open(image_path).convert("RGB")

        # DonutProcessor로 이미지 전처리 -> pixel_values
        pixel_values = self.processor(
            image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # DonutProcessor의 tokenizer로 ground truth 토큰화
        target_ids = self.processor.tokenizer(
            ground_truth,
            add_special_tokens=False,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
         # PAD 토큰을 -100으로 바꿔 loss 계산 시 무시되도록 설정
        target_ids[target_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": pixel_values,
            "labels": target_ids
        }