from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# 学習用データの読み込み
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

data = pd.read_json('./new_training.json')
train, valid = train_test_split(data, test_size=0.2)
ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)

dataset = DatasetDict({
    "train": ds_train,
    "validation": ds_valid,
})
