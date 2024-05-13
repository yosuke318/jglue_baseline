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

# デバイス判定
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# 変換関数
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=450,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt").to(device)

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


# 変換
tokenized_data = dataset.map(preprocess_function, batched=True)

# モデル取得
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("roberta-base").to(device)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=200,
    weight_decay=0.01,
    load_best_model_at_end=True, # 終了時に一番良かったモデルを使う
)

from transformers import default_data_collator
data_collator = default_data_collator

from transformers import Trainer, EarlyStoppingCallback

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()
