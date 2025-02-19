import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import onnx

# 파인튜닝한 Donut 모델 경로 또는 허브 모델명을 지정합니다.
model_name = "epoch-023_step-01200_ED-0.0554"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.eval()  # 평가 모드로 전환

# 입력 이미지 준비: processor에서 기대하는 이미지 크기로 (보통 processor.image_processor.size)
width, height = processor.image_processor.size["width"], processor.image_processor.size["height"]
dummy_image = Image.new("RGB", (width, height), (255, 255, 255))  # 흰색 배경의 더미 이미지

# processor를 통해 pixel_values 생성 (shape: [1, 3, H, W])
inputs = processor(dummy_image, return_tensors="pt")

# Donut는 decoder에 특별한 prompt (예: "<s_doc>")를 사용합니다.
# 일반적으로 tokenizer의 cls_token_id가 해당 토큰에 해당합니다.
task_prompt = "<s_cord-v2>"
dummy_decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

# ONNX 내보내기
onnx_model_path = "donut_model.onnx"
torch.onnx.export(
    model,
    (inputs["pixel_values"], dummy_decoder_input_ids),
    onnx_model_path,
    input_names=["pixel_values", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "decoder_input_ids": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=20  # 필요에 따라 버전 조정
)
print("Donut ONNX 모델이 성공적으로 내보내졌습니다.")

onnx_model = onnx.load(onnx_model_path)
modified = False

for node in onnx_model.graph.node:
    if node.op_type == "ScatterND":
        for attr in node.attribute:
            if attr.name == "reduction":
                print("수정 전 ScatterND reduction:", attr.s.decode("utf-8"))
                attr.s = b"add"
                print("수정 후 ScatterND reduction:", attr.s.decode("utf-8"))
                modified = True

if not modified:
    print("ScatterND 노드에서 수정할 'reduction' 속성을 찾지 못했습니다.")

# 5. 수정된 ONNX 모델 저장
modified_onnx_model_path = "donut_model_modified.onnx"
onnx.save(onnx_model, modified_onnx_model_path)
print(f"수정된 ONNX 모델이 {modified_onnx_model_path}에 저장되었습니다.")