from transformers.trainer_utils import set_seed
from transformers import (BatchEncoding, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification)

import numpy as np
from scipy.stats import pearsonr, spearmanr

from pprint import pprint
from datasets import load_dataset

# 乱数シードを42に固定
set_seed(42)

# Hugging Face Hub上のllm-book/JGLUEのリポジトリからJSTSのデータを読み込む
train_dataset = load_dataset(
    "shunk031/JGLUE", name="JSTS", split="train"
)
valid_dataset = load_dataset(
    "shunk031/JGLUE", name="JSTS", split="validation"
)

# Hugging Face Hub上のモデル名を指定
model_name = "cl-tohoku/bert-base-japanese-v3"
# モデル名からトークナイザを読み込む
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess_text_pair_classification(
    example: dict[str, str | int]
) -> BatchEncoding:
    """文ペア関係予測の事例をトークナイズし、IDに変換"""
    # 出力は"input_ids", "token_type_ids", "attention_mask"をkeyとし、
    # list[int]をvalueとするBatchEncodingオブジェクト
    encoded_example = tokenizer(
        example["sentence1"], example["sentence2"], max_length=128
    )

    # BertForSequenceClassificationのforwardメソッドが
    # 受け取るラベルの引数名に合わせて"labels"をキーにする
    encoded_example["labels"] = example["label"]
    return encoded_example


# train, valid, testデータをencodeする
encoded_train_dataset = train_dataset.map(
    preprocess_text_pair_classification,
    remove_columns=train_dataset.column_names,
)

encoded_valid_dataset = valid_dataset.map(
    preprocess_text_pair_classification,
    remove_columns=valid_dataset.column_names,
)


# ミニバッチ構築
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

batch_inputs = data_collator(encoded_train_dataset[0:4])


transformers_model_name = "cl-tohoku/bert-base-japanese-v3"

model = AutoModelForSequenceClassification.from_pretrained(
    transformers_model_name,
    num_labels=1,
    problem_type="regression",
)

result_path = './tasks/jsts/result'

training_args = TrainingArguments(
    output_dir=result_path,  # 結果の保存フォルダ
    auto_find_batch_size=True,  # 自動で調整
    # per_device_train_batch_size=32,  # 訓練時のバッチサイズ
    # per_device_eval_batch_size=32,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type="linear",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=1,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="spearmanr",  # 最良のモデルを決定する評価指標
    # fp16=True,  # 自動混合精度演算の有効化(cudaかNPUのみ使える)
)


def compute_correlation_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """
    予測スコアと正解スコアからピアソン相関係数とスピアマン相関係数を計算
    """
    predictions, labels = eval_pred
    predictions = predictions.squeeze(1)
    return {
        "pearsonr": pearsonr(predictions, labels).statistic,
        "spearmanr": spearmanr(predictions, labels).statistic,
    }


trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_correlation_metrics,
)
trainer.train()


# 検証セットでモデルを評価
eval_metrics = trainer.evaluate(encoded_valid_dataset)
pprint(eval_metrics)

