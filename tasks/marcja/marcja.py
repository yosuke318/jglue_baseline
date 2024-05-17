from transformers import (AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification,
                          Trainer, EarlyStoppingCallback, default_data_collator,
                          pipeline)
import torch
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import numpy as np

from tasks.marcja.load_marcja_data import load_marc_data

train_dataset, valid_dataset = load_marc_data()

# データセットをDataFrameに変換
df_train = pd.DataFrame(train_dataset)
df_valid = pd.DataFrame(valid_dataset)

# 最初の100行を抽出
df_train = df_train.head(80)
df_valid = df_valid.head(20)

# dfをdataset型へ戻す
ds_train = Dataset.from_pandas(df_train)
ds_valid = Dataset.from_pandas(df_valid)

dataset = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
})

dataset.map(lambda example: {'label': 1 if example['label'] == 'positive' else 0})

# 東北大モデルのトークンナイザー
tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')


def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_ds = tokenized_datasets['train'].shuffle(seed=51)  # シャッフルするとデータの偏りがなくなるらしい
small_valid_ds = tokenized_datasets['valid'].shuffle(seed=51)

# 東北大モデル
model = AutoModelForSequenceClassification.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking',
                                                           num_labels=2)

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='outputs',
    evaluation_strategy='epoch',
    auto_find_batch_size=True,
    learning_rate=5e-05,
    num_train_epochs=4,
    warmup_ratio=0.1,
    save_steps=5000
)


