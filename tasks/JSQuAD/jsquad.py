from transformers import (AutoTokenizer, TrainingArguments, AutoModelForQuestionAnswering,
                          Trainer, EarlyStoppingCallback, default_data_collator,
                          pipeline)
import torch
import pandas as pd

from datasets import Dataset, DatasetDict

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

train = pd.read_json('./new_training.json')
valid = pd.read_json('./new_training.json')

ds_train = Dataset.from_pandas(train)
ds_valid = Dataset.from_pandas(valid)

dataset = DatasetDict({
    "train": ds_train,
    "validation": ds_valid,
})

# GPUが使えるか判定(できれば実行環境はGPUが良い)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForQuestionAnswering.from_pretrained("roberta-base").to(device)  # GPUにのせる



def preprocess_function(examples):
    """
    # 前処理の関数(preprocess_functionを使用：https://huggingface.co/docs/transformers/v4.19.2/en/tasks/question_answering)
    # note:QAタスクの場合、正解の文字の始まりと終わりの位置を示す必要がある
    :param examples:
    :return: inputs["start_positions"], inputs["end_positions"]
    """
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

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # epoch数
    weight_decay=0.01,
    load_best_model_at_end=True  # 終了時に一番良かったモデルを使う
)

data_collator = default_data_collator

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

# 学習したモデルで推論
pipe = pipeline(task='question-answering',
                model=trainer.model,
                tokenizer=trainer.tokenizer)

result = pipe(question="多国籍企業において、防衛で有名なのはなにUSA？",
              context="多国籍企業 [SEP] 公企業や地域に偏りのあるものも紹介している。4大会計事務所も有名。防衛ではブラックウォーターUSA",
              top_k=2,
              handle_impossible_answer=False,
              align_to_words=False)

print(result)
