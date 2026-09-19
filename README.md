# JGLUE ベースライン

日本語言語理解ベンチマーク [JGLUE](https://github.com/yahoojapan/JGLUE) の各タスクについて、Hugging Face Transformers を使ったベースラインモデルを作成するリポジトリです。

## 対応タスク

| タスク | 内容 | 使用モデル | スクリプト | 使用データ |
| --- | --- | --- | --- | --- |
| JSTS | 文ペアの意味的類似度（回帰） | `cl-tohoku/bert-base-japanese-v3` | [tasks/jsts/jsts.py](tasks/jsts/jsts.py) | [shunk031/JGLUE](https://huggingface.co/datasets/shunk031/JGLUE)（一部抜粋） |
| JSTS | 同上（RoBERTa を使ったサンプル実装） | `roberta-base` | [tasks/jsts/jsts_sample.py](tasks/jsts/jsts_sample.py) | [jsts-v1.1](https://github.com/yahoojapan/JGLUE/tree/main/datasets/jsts-v1.1)（一部抜粋） |
| JSQuAD | 抽出型質問応答 | `roberta-base` | [tasks/JSQuAD/jsquad.py](tasks/JSQuAD/jsquad.py) | [jsquad-v1.1](https://github.com/yahoojapan/JGLUE/tree/main/datasets/jsquad-v1.1)（一部抜粋） |
| MARC-ja | 商品レビューの感情分類（2値） | `tohoku-nlp/bert-base-japanese` | [tasks/marcja/marcja.py](tasks/marcja/marcja.py) | [shunk031/JGLUE](https://huggingface.co/datasets/shunk031/JGLUE)（一部抜粋） |

> [!NOTE]
> 動作確認を目的としたベースラインのため、各タスクとも学習・検証データは一部のみ（例: 学習 80 件 / 検証 20 件）を使い、エポック数も 1 にしています。

## ディレクトリ構成

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── tasks/
    ├── jsts/
    │   ├── jsts.py              # 東北大 BERT による JSTS
    │   ├── jsts_sample.py       # RoBERTa による JSTS（サンプル）
    │   ├── training.json        # 学習データ（抜粋）
    │   └── validation.json      # 検証データ（抜粋）
    ├── JSQuAD/
    │   ├── convert.py           # SQuAD 形式 → 学習用フラット形式への変換
    │   ├── jsquad.py            # RoBERTa による JSQuAD
    │   ├── training.json        # 学習データ（抜粋）
    │   ├── validation.json      # 検証データ（抜粋）
    │   └── prediction.json      # 推論用サンプル
    └── marcja/
        ├── load_marcja_data.py  # MARC-ja データの取得・CSV 出力
        └── marcja.py            # 東北大 BERT による MARC-ja
```

## 環境構築

### Docker を使う場合

```bash
docker compose up -d --build
```

```bash
docker exec -it jsts_container bash
```

### ローカルで実行する場合

Python 3.11 を想定しています。

```bash
pip install -r requirements.txt
```

GPU（CUDA）が使える環境では自動的に GPU で学習します。

## 実行方法

スクリプトごとに想定しているカレントディレクトリが異なるので注意してください。

### JSTS

リポジトリのルートで実行します。データは Hugging Face Hub から自動で取得します。

```bash
python tasks/jsts/jsts.py
```

RoBERTa 版のサンプルは `tasks/jsts` ディレクトリ内で実行します。

```bash
cd tasks/jsts && mkdir -p models && python jsts_sample.py
```

### JSQuAD

1. `convert.py` で SQuAD 形式のデータを学習用の形式（`new_training.json` / `new_validation.json`）に変換します。現状は入力・出力ファイル名がスクリプト内に直書きされているため、学習用と検証用でそれぞれ書き換えて実行してください。

   ```bash
   cd tasks/JSQuAD && python convert.py
   ```

2. リポジトリのルートで学習を実行します。

   ```bash
   mkdir -p tasks/JSQuAD/models && python tasks/JSQuAD/jsquad.py
   ```

### MARC-ja

リポジトリのルートからモジュールとして実行します。データは Hugging Face Hub から自動で取得します。

```bash
mkdir -p tasks/marcja/models tasks/marcja/result && python -m tasks.marcja.marcja
```

データの中身を確認したい場合は、`load_marcja_data.py` を直接実行すると `tasks/marcja/` に CSV（`ds_train.csv` / `ds_valid.csv`）が出力されます（`marcja.py` から import された場合は出力されません）。

```bash
python -m tasks.marcja.load_marcja_data
```

## 出力

| タスク | 学習結果（チェックポイント） | 保存モデル |
| --- | --- | --- |
| JSTS（`jsts.py`） | `tasks/jsts/result/` | ―（評価指標を標準出力に表示） |
| JSTS（`jsts_sample.py`） | ― | `tasks/jsts/models/roberta_sequence_classification_model_<commit_id>.pt` |
| JSQuAD | `tasks/JSQuAD/result/` | `tasks/JSQuAD/models/jsquad_model_<commit_id>.pt` |
| MARC-ja | `tasks/marcja/result/` | `tasks/marcja/models/marcja_model_<commit_id>.pt` |

保存モデルのファイル名には実行時点の Git のコミット ID が付与されるため、どのコードで学習したモデルかを追跡できます。MARC-ja はテストデータに対する予測結果と評価指標（accuracy / precision / recall / F1）も `tasks/marcja/result/marcja_<commit_id>.csv` に出力します。

## 評価指標

| タスク | 指標 |
| --- | --- |
| JSTS | ピアソン相関係数・スピアマン相関係数 |
| JSQuAD | 検証 loss |
| MARC-ja | accuracy / precision / recall / F1 |

## 参考

- [JGLUE (yahoojapan/JGLUE)](https://github.com/yahoojapan/JGLUE)
- [shunk031/JGLUE (Hugging Face Datasets)](https://huggingface.co/datasets/shunk031/JGLUE)
- [Hugging Face Transformers: Question answering](https://huggingface.co/docs/transformers/tasks/question_answering)
