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
df_test = df_valid.tail(20)

# dfをdataset型へ戻す
ds_train = Dataset.from_pandas(df_train)
ds_valid = Dataset.from_pandas(df_valid)
ds_test = Dataset.from_pandas(df_test)

dataset = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
    "test": ds_test
})

dataset.map(lambda example: {'label': 1 if example['label'] == 'positive' else 0})

# ロベルトのトークンナイザー
# tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')


def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_ds = tokenized_datasets['train'].shuffle(seed=51)  # シャッフルするとデータの偏りがなくなるらしい
small_valid_ds = tokenized_datasets['valid'].shuffle(seed=51)
small_test_ds = tokenized_datasets['test'].shuffle(seed=51)

# ロベルト
# GPUが使えるか判定(できれば実行環境はGPUが良い)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                           num_labels=2).to(device)

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='./result',
    evaluation_strategy='epoch',
    auto_find_batch_size=True,
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    learning_rate=5e-05,
    num_train_epochs=1,
    warmup_ratio=0.1,
    save_steps=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_ds,
    eval_dataset=small_valid_ds,
    compute_metrics=compute_metrics
)

trainer.train()

df_test_list = df_test["sentence"].tolist()

test_tokenized = tokenizer(df_test_list, return_tensors="pt", padding=True, truncation=True, max_length=512)

outputs = trainer.predict(test_dataset=small_test_ds)
predicted_labels = outputs.predictions.argmax(axis=1)

# 予測結果を元のDataFrameに結合
df_test["predicted_label"] = predicted_labels

df_test.to_csv('./marcja.csv')
