from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
from PIL import Image

# 예시로 사전학습된 모델 불러오기 (모델 이름은 원하는 것으로 대체)
model_name = "epoch-023_step-01200_ED-0.0554"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)
model.eval()


# encoder와 decoder 분리
encoder = model.encoder
decoder = model.decoder

# Encoder 변환

dummy_image = Image.new("RGB", (224, 224), (255, 255, 255))  # 흰색 배경의 더미 이미지
inputs = processor(dummy_image, return_tensors="pt")

torch.onnx.export(
    encoder,
    inputs["pixel_values"],
    "encoder.onnx",
    input_names=["pixel_values"],
    output_names=["encoder_output"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "encoder_output": {0: "batch_size"}
    },
    opset_version=12
)

print('encoder complete!')
# input_ids = torch.ones((1, 1), dtype=torch.int32)
# test_inputs = torch.randn(1,3,1280,960)
# dummy_encoder_outputs = encoder(test_inputs).last_hidden_state

# dummy_input_decoder = dict(input_ids=input_ids, encoder_hidden_states=dummy_encoder_outputs, use_cache=True, return_dict=True)
# a = decoder(**dummy_input_decoder)



test_inputs = torch.randn(1,3,1280,960)

task_prompt = "<s_cord-v2>"
dummy_decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
# dummy_encoder_outputs = encoder(inputs["pixel_values"]).last_hidden_state
dummy_encoder_outputs = encoder(test_inputs).last_hidden_state
dummy_input_decoder = dict(input_ids=dummy_decoder_input_ids, encoder_hidden_states=dummy_encoder_outputs)

# ONNX로 변환
torch.onnx.export(
    decoder,
    (dummy_decoder_input_ids, dummy_input_decoder),
    "decoder.onnx",
    input_names=["input_ids", "encoder_outputs"],
    output_names=["decoder_outputs"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "encoder_outputs": {0: "batch_size", 1: "encoder_seq_length"},
        "decoder_outputs": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=14
)

print('decoder complete!')