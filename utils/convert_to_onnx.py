from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import AutoTokenizer, DonutProcessor

model_checkpoint = "epoch-023_step-01200_ED-0.0554"
save_directory = "onnx"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
processor = DonutProcessor.from_pretrained(model_checkpoint)
ort_model = ORTModelForVision2Seq.from_pretrained(model_checkpoint, export=True)

ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"ONNX 모델이 저장되었습니다 : {save_directory}")