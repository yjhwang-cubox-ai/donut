from transformers import DonutProcessor
from optimum.onnxruntime import ORTModelForVision2Seq
from PIL import Image
import time

start_time = time.time()

processor = DonutProcessor.from_pretrained("onnx_donut")
model = ORTModelForVision2Seq.from_pretrained("onnx_donut")

img_name = "test_images/test-exam.jpg"
image = Image.open(img_name)
inputs = processor(image, return_tensors="pt")

task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
inputs["decoder_input_ids"] = decoder_input_ids

gen_tokens = model.generate(**inputs)
outputs = processor.batch_decode(gen_tokens, skip_special_tokens=True)

print(outputs)

print(f"inference time: {time.time() - start_time}")