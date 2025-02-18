from transformers import DonutProcessor, GenerationConfig
import onnxruntime as ort
import numpy as np
from PIL import Image
from tqdm import tqdm

model_name = "epoch-023_step-01200_ED-0.0554"
processor = DonutProcessor.from_pretrained(model_name)

# ONNX Runtime 세션 생성
encoder_session = ort.InferenceSession("encoder.onnx")
decoder_session = ort.InferenceSession("decoder.onnx")

# 1. Encoder 추론
image_path = "test-exam.jpg"  # 추론에 사용할 문서 이미지 경로
image = Image.open(image_path).convert("RGB")
inputs = processor(image, return_tensors="np")
encoder_inputs = {"pixel_values": inputs['pixel_values']}
encoder_out = encoder_session.run(None, encoder_inputs)[0]


# # 4. 필요한 토큰 id 추출
pad_token_id = processor.tokenizer.pad_token_id
eos_token_id = processor.tokenizer.eos_token_id
unk_token_id = processor.tokenizer.unk_token_id
# generation_config.json 파일에서 max_length 정보 로드
gen_config = GenerationConfig.from_pretrained(model_name)
max_length = gen_config.max_length

# 2. Decoder 추론
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="np")["input_ids"]

generated_ids = decoder_input_ids[0].tolist()

for _ in tqdm(range(max_length - len(generated_ids))):
    current_decoder_input_ids = np.array([generated_ids], dtype=np.int64)
    inputs = {
        "input_ids": current_decoder_input_ids,
        "encoder_outputs": encoder_out
    }
    outputs = decoder_session.run(None, inputs)
    logits = outputs[0]  # shape: [1, seq_len, vocab_size]
    
    # 마지막 토큰에 대한 logits
    next_token_logits = logits[0, -1, :].copy()
    
    # unk 토큰의 확률을 낮춰 선택되지 않도록 함 (argmax 전에 적용)
    next_token_logits[unk_token_id] = -1e9
    next_token_id = int(np.argmax(next_token_logits))
    
    generated_ids.append(next_token_id)
    
    # EOS 토큰이 생성되면 종료하여 중복된 인덱스 업데이트를 방지
    if next_token_id == eos_token_id:
        break

print(generated_ids)
sequences = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
decoded_text = processor.tokenizer.decode(sequences, skip_special_tokens=True)
print("문서 파싱 결과:", decoded_text)














# decoder_inputs = {
#     "input_ids": decoder_input_ids,
#     "encoder_outputs": encoder_out
# }
# logits = decoder_session.run(None, decoder_inputs)[0]
# next_token_logits = logits[0, -1, :].copy()
# next_token_logits[unk_token_id] = -1e9
# next_token_id = int(np.argmax(next_token_logits))


# print(next_token_id)