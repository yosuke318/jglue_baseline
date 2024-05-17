from pprint import pprint
from datasets import load_dataset
import pandas as pd


def load_marc_data():
    """
    # Hugging Face Hub上のshunk031/JGLUEのリポジトリから
    # MARC-jaのデータを読み込む
    # 読込先：https://huggingface.co/datasets/shunk031/JGLUE
    :return:
    """
    train_dataset = load_dataset(
        "shunk031/JGLUE", name="MARC-ja", split="train"
    )
    valid_dataset = load_dataset(
        "shunk031/JGLUE", name="MARC-ja", split="validation"
    )
    return train_dataset, valid_dataset


train_dataset, valid_dataset = load_marc_data()

# データ確認用
df = pd.DataFrame(train_dataset)
df.to_csv('ds_train.csv', index=False)

df = pd.DataFrame(valid_dataset)
df.to_csv('ds_valid.csv', index=False)