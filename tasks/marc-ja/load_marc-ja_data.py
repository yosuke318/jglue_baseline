from pprint import pprint
from datasets import load_dataset
import pandas as pd

# Hugging Face Hub上のllm-book/JGLUEのリポジトリから
# MARC-jaのデータを読み込む
# 読込先：https://huggingface.co/datasets/shunk031/JGLUE
train_dataset = load_dataset(
    "shunk031/JGLUE", name="MARC-ja", split="train"
)
valid_dataset = load_dataset(
    "shunk031/JGLUE", name="MARC-ja", split="validation"
)

df = pd.DataFrame(train_dataset)
df.to_csv('ds_train.csv', index=False)

df = pd.DataFrame(valid_dataset)
df.to_csv('ds_valid.csv', index=False)