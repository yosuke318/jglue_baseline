import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import json
import git
import random
import numpy as np

# 乱数生成器のシードを設定
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# データの準備
class CustomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Dataloader読み取り用のデータを返す。
        :param idx:
        :return:
        """
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.max_len,
                                  return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
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
val_data = json.load(valid_json)['valid']

train_sentences = [data['sentence1'] + ' ' + data['sentence2'] for data in train_data]
train_labels = [data['label'] for data in train_data]

val_sentences = [data['sentence1'] + ' ' + data['sentence2'] for data in val_data]
val_labels = [data['label'] for data in val_data]

# トークナイザーの準備
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_dataset = CustomDataset(train_sentences, train_labels, tokenizer, MAX_LEN)
val_dataset = CustomDataset(val_sentences, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# robertaモデルの準備
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
model.train()

# オプティマイザーの設定
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 学習ループ
for epoch in range(EPOCHS):
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # バリデーション
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Loss: {val_loss / len(val_loader)}")

# 以下は推論
# 推論用のデータを用意する
new_data = [
    {
        "sentence1": "林の中を1台の自転車が走行しています。",
        "sentence2": "林の中で1台の自転車が走行しています。",
    },
    {
        "sentence1": "猫が庭で寝ている",
        "sentence2": "ゴミが落ちている",
    }
]

# データをトークン化し、PyTorchのテンソルに変換する
inputs = []
for example in new_data:
    encoding = tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
    inputs.append({
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
    })

# 推論を実行する
model.eval()
with torch.no_grad():
    for example in inputs:
        input_ids = example['input_ids'].unsqueeze(0)  # バッチサイズ1の次元を追加
        attention_mask = example['attention_mask'].unsqueeze(0)  # バッチサイズ1の次元を追加
        outputs = model(input_ids, attention_mask=attention_mask)
        print("Prediction:", outputs.logits.item())


# 現在のリポジトリのコミットIDを取得
repo = git.Repo(search_parent_directories=True)
commit_id = repo.head.object.hexsha

# モデルを保存するパス
save_path = f"models/roberta_sequence_classification_model_{commit_id}.pt"

# モデルを保存
torch.save(model.state_dict(), save_path)
