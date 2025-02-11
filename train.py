import os
import json
import torch
from datasets import load_dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from PIL import Image

# 1. 모델과 프로세서 로드
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base", cache_dir='./hub/processor', legacy=False)
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", cache_dir='./hub/model')

# 2. 데이터셋 로드
train_dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
valid_dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")

# 3. 전처리 함수 정의
def preprocess(example):
    # PIL Image가 RGB 모드가 아니라면 변환
    image = example["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    # 이미지를 processor를 통해 tensor로 변환 (pixel_values)
    pixel_values = processor(image, return_tensors="pt").pixel_values[0]  # [C, H, W] tensor
    
    # 정답(annotation) 전처리  
    # ※ CORD 데이터셋의 정답 필드명이 "ground_truth"로 되어있다고 가정함.
    # 실제 데이터셋에 따라 "annotation" 등으로 변경 필요.
    target = json.dumps(example["ground_truth"], ensure_ascii=False)
    # tokenizer를 이용해 정답 텍스트를 토큰화 (special token 없이)
    labels = processor.tokenizer(target, add_special_tokens=False).input_ids

    example["pixel_values"] = pixel_values
    example["labels"] = labels
    return example

# 4. 데이터셋에 전처리 함수 적용 (불필요한 원본 컬럼은 제거)
train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names, num_proc=4)
valid_dataset = valid_dataset.map(preprocess, remove_columns=valid_dataset.column_names, num_proc=4)

# 5. 데이터 배치 구성 (collator)
def collate_fn(batch):
    # 이미지 tensor는 동일한 사이즈이므로 stack
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    # labels는 길이가 다를 수 있으므로 tokenizer의 pad 기능 활용
    labels = [item["labels"] for item in batch]
    labels = processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt").input_ids
    return {"pixel_values": pixel_values, "labels": labels}

training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-cord",
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=100,
    save_steps=1000,
    learning_rate=5e-5,
    predict_with_generate=True,
    fp16=True,  # GPU 환경에서 fp16 사용 가능할 경우
    push_to_hub=False,
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # 패딩 토큰을 제거한 후 디코딩
    decoded_preds = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    # 단순 정확도 계산 (문자열 단위로 비교)
    accuracy = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)
    return {"accuracy": accuracy}

# 7. Trainer 객체 생성
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,  # 필요하지 않으면 제거 가능
)

# 8. 모델 파인튜닝 시작
trainer.train()

# 9. (선택) 평가 및 모델 저장
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

trainer.save_model("./donut-cord-final")