from typing import Type

import torch
from transformers import RobertaTokenizer, RobertaForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
import json
import git
import random
import numpy as np
import torch.nn as nn

# 乱数生成器のシードを設定
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# データの準備
class CustomDataset(Dataset):
    def __init__(self, qas, labels, tokenizer, max_len):
        self.qas = qas
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.qas)

    def __getitem__(self, idx):
        """
        Dataloader読み取り用のデータを返す。
        :param idx:
        :return:
        """
        qas = self.qas[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(qas, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }


# ハイパーパラメータ
MAX_LEN = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3

# 学習データの読み込み
train_json = open('training.json', 'r')
train_data = json.load(train_json)['train']

# 検証データの読み込み
valid_json = open('validation.json', 'r')
valid_data = json.load(valid_json)['valid']

# trainデータを生成
train_qa = []
train_labels = []

for item in train_data:
    context = item['paragraphs'][0]['context']
    for qa in item['paragraphs'][0]['qas']:
        question = qa['question']
        answer_text = qa['answers'][0]['text']
        train_qa.append(question + ' ' + context)
        train_labels.append(answer_text)


# validデータを生成
valid_qa = []
valid_labels = []

for item in valid_data:
    context = item['paragraphs'][0]['context']
    for qa in item['paragraphs'][0]['qas']:
        question = qa['question']
        answer_text = qa['answers'][0]['text']
        valid_qa.append(question + ' ' + context)
        valid_labels.append(answer_text)

# トークナイザーの準備
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_dataset = CustomDataset(train_qa, train_labels, tokenizer, MAX_LEN)
val_dataset = CustomDataset(valid_qa, valid_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# robertaモデルの準備
model = RobertaForQuestionAnswering.from_pretrained('roberta-base')
model.train()

# オプティマイザーの設定
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# 学習ループ
for epoch in range(EPOCHS):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = torch.tensor([tokenizer.encode(answer, add_special_tokens=False)[0] for answer in batch['labels']], dtype=torch.long).to(input_ids.device)

        optimizer.zero_grad()

        outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)

        optimizer.step()

#     # バリデーション
#     val_loss = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             input_ids = batch['input_ids']
#             attention_mask = batch['attention_mask']
#             labels = batch['labels']
#
#             outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             val_loss += outputs.loss.item()
#
#     print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss / len(val_loader)}")
#
# # 以下は推論
# # 推論データの読み込み
# pred_json = open('prediction.json', 'r')
# pred_data = json.load(train_json)['qas']
#
# # データをトークン化し、PyTorchのテンソルに変換する
# inputs = []
# for example in pred_data:
#     encoding = tokenizer(example["question"], example["answers"], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
#     inputs.append({
#         'input_ids': encoding['input_ids'].flatten(),
#         'attention_mask': encoding['attention_mask'].flatten(),
#     })
#
# # 推論を実行する
# model.eval()
# with torch.no_grad():
#     for example in inputs:
#         input_ids = example['input_ids'].unsqueeze(0)  # バッチサイズ1の次元を追加
#         attention_mask = example['attention_mask'].unsqueeze(0)  # バッチサイズ1の次元を追加
#         outputs = model(input_ids, attention_mask=attention_mask)
#         print("Prediction:", outputs.logits.item())