import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image
import numpy as np

# 1. 모델과 프로세서 로드
model_name = "epoch-023_step-01200_ED-0.0554"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)

# 2. ONNX 내보내기를 위한 wrapper 클래스 정의
class DonutONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, decoder_input_ids):
        # 모델의 forward()는 Seq2SeqLMOutput(여러 값을 포함)을 리턴하므로, 여기서는 logits만 반환
        outputs = self.model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        return outputs.logits

onnx_model = DonutONNXWrapper(model)

# 3. 더미 입력 생성
# 임의의 이미지 생성 (예: 800x800 크기의 랜덤 RGB 이미지)
dummy_image = Image.fromarray(np.uint8(np.random.rand(800, 800, 3) * 255))

inputs = processor(dummy_image, return_tensors="pt")
pixel_values = inputs["pixel_values"]  # shape: [1, 3, H, W]

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

# 4. ONNX로 변환
torch.onnx.export(
    onnx_model,
    (pixel_values, decoder_input_ids),   # 모델에 전달할 입력 튜플
   "donut.onnx",                          # 출력 ONNX 파일명
    input_names=["pixel_values", "decoder_input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "decoder_input_ids": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=20,  # 사용 가능한 ONNX opset 버전 (필요에 따라 조정)
)

print("ONNX 모델이 'donut.onnx'로 성공적으로 내보내졌습니다!")
