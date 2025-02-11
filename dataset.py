from datasets import load_dataset
import json

ds = load_dataset("naver-clova-ix/cord-v2", cache_dir='./data')

train_dataset = load_dataset("naver-clova-ix/cord-v2", split="train")
valid_dataset = load_dataset("naver-clova-ix/cord-v2", split="validation")
test_dataset  = load_dataset("naver-clova-ix/cord-v2", split="test")

print(ds)