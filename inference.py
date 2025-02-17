import re
import json
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
from datasets import load_dataset
import torch
from PIL import Image
from tqdm import tqdm

processor = DonutProcessor.from_pretrained("logs/Donut-finetuning/finetunig-cord-TEST/epoch-005_step-02400_ED-0.0631")
model = VisionEncoderDecoderModel.from_pretrained("logs/Donut-finetuning/finetunig-cord-TEST/epoch-005_step-02400_ED-0.0631")


image = Image.open("test-exam-train.jpg")
pixel_values = processor(image, return_tensors="pt").pixel_values


task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

outputs = model.generate(
    pixel_values.to(device),
    decoder_input_ids=decoder_input_ids.to(device),
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=False,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,)
    # output_scores=True,)



sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
print(sequence)

sequence = processor.token2json(sequence)

with open("result.json", "w", encoding="utf-8") as json_file:
    json.dump(sequence, json_file, ensure_ascii=False, indent=4)