from transformers import (AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification,
                          Trainer, EarlyStoppingCallback)
import torch
import os
import git
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tasks.marcja.load_marcja_data import load_marc_data

train_dataset, valid_dataset = load_marc_data()

# データセットをDataFrameに変換
df_train = pd.DataFrame(train_dataset)
df_valid = pd.DataFrame(valid_dataset)

# 最初の100行を抽出
df_train = df_train.head(80)
df_valid = df_valid.head(20)
df_test = df_valid.tail(20)  # testデータはvalidから取得する

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

# 東北大のトークンナイザー
# tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese-whole-word-masking')
# https://huggingface.co/tohoku-nlp/bert-base-japanese
tokenizer = AutoTokenizer.from_pretrained('tohoku-nlp/bert-base-japanese')


def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=512)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_ds = tokenized_datasets['train'].shuffle(seed=51)  # シャッフルするとデータの偏りがなくなるらしい
small_valid_ds = tokenized_datasets['valid'].shuffle(seed=51)
small_test_ds = tokenized_datasets['test'].shuffle(seed=51)

# 東北大
# GPUが使えるか判定(できれば実行環境はGPUが良い)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained('tohoku-nlp/bert-base-japanese',
                                                           num_labels=2).to(device)

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir='./tasks/marcja/result',
    evaluation_strategy='epoch',
    auto_find_batch_size=True,  # 自動で調整
    # per_device_train_batch_size=16,
    # per_device_eval_batch_size=16,
    learning_rate=5e-05,
    num_train_epochs=1,
    warmup_ratio=0.1,
    save_steps=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_ds,
    eval_dataset=small_valid_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

df_test_list = df_test["sentence"].tolist()

test_tokenized = tokenizer(df_test_list, return_tensors="pt", padding=True, truncation=True, max_length=512)

outputs = trainer.predict(test_dataset=small_test_ds)
predicted_labels = outputs.predictions.argmax(axis=1)

# 予測結果をデータフレームに追加して保存
df_test["predicted_label"] = predicted_labels

true_labels = df_test["label"].tolist()

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')

df_test["accuracy"] = accuracy
df_test["precision"] = precision
df_test["recall"] = recall
df_test["f1"] = f1

current_directory = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(current_directory, 'models')

os.makedirs(model_path, exist_ok=True)

# 現在のリポジトリのコミットIDを取得
repo = git.Repo(search_parent_directories=True)
commit_id = repo.head.object.hexsha

# モデルを保存するパス（任意のパスでOK）
save_path = f"./{model_path}/marcja_model_{commit_id}.pt"
df_test.to_csv(f'./tasls/marcja/result/marcja_{commit_id}.csv')

torch.save(model.state_dict(), save_path)
