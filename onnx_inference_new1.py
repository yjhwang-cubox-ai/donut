import re
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

# processor와 원본 모델 (토크나이저 정보 용) 로드
processor = DonutProcessor.from_pretrained('epoch-023_step-01200_ED-0.0554')
model = VisionEncoderDecoderModel.from_pretrained('epoch-023_step-01200_ED-0.0554')

# ONNX Runtime 세션 생성
encoder_session = ort.InferenceSession("donut_encoder.onnx")
decoder_session = ort.InferenceSession("donut_decoder.onnx")

# --- 이미지 전처리 ---
image = Image.open("test-exam.jpg")
pixel_values = processor(image, return_tensors="pt").pixel_values  # (1, 3, H, W)
np_pixel_values = pixel_values.numpy()

# --- Encoder 추론 ---
encoder_inputs = {"pixel_values": np_pixel_values}
encoder_hidden_states = encoder_session.run(None, encoder_inputs)[0]

# --- Decoder 초기 입력 (task prompt) ---
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
np_decoder_input_ids = decoder_input_ids.numpy()  # shape: (1, seq_length)

# 최대 생성 길이 및 eos token id 설정
max_length = model.decoder.config.max_position_embeddings
eos_token_id = processor.tokenizer.eos_token_id
pad_token_id = processor.tokenizer.pad_token_id

# --- Greedy decoding (attention mask 포함) ---
while True:
    # decoder_attention_mask: pad가 아닌 토큰은 1, pad이면 0
    np_decoder_attention_mask = (np_decoder_input_ids != pad_token_id).astype(np.int64)
    
    decoder_inputs = {
        "decoder_input_ids": np_decoder_input_ids,
        "decoder_attention_mask": np_decoder_attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
    }
    # decoder ONNX 세션 실행: logits shape은 (batch_size, decoder_seq_length, vocab_size)
    logits = decoder_session.run(None, decoder_inputs)[0]
    
    # 마지막 토큰의 logits에서 argmax
    next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
    next_token = np.argmax(next_token_logits, axis=-1)  # (batch_size,)
    
    # 새 토큰을 decoder 입력에 추가
    np_decoder_input_ids = np.concatenate([np_decoder_input_ids, next_token[:, None]], axis=1)
    
    # EOS 토큰 도달 혹은 최대 길이 초과 시 종료
    if next_token[0] == eos_token_id or np_decoder_input_ids.shape[1] >= max_length:
        break

# --- 최종 시퀀스 후처리 ---
sequence = processor.tokenizer.decode(np_decoder_input_ids[0], skip_special_tokens=True)
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
print("생성 결과:", sequence)

# --- 결과 JSON 저장 ---
result = processor.token2json(sequence)
with open("result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)