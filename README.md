# SFTトレーニング環境セットアップと実行

このプロジェクトは [llm-jp/llm-jp-sft](https://github.com/llm-jp/llm-jp-sft) をベースに、現在のバージョンに合わせたカスタマイズがされています。

## 環境セットアップ

### 1. 環境の作成と有効化
```bash
conda create -n sft python=3.11 -y
conda activate sft
```

### 2. 必要なパッケージのインストール
```bash
cd document/sft
pip install -r requirements.in
```

### 3. 環境変数の設定
`{path/to/your/miniconda3}` または `{path/to/your/anaconda3}` を実際のパスに置き換えてください。
```bash
export LD_LIBRARY_PATH={path/to/your/miniconda3}/envs/sft/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH={path/to/your/anaconda3}/envs/sft/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

### 4. 追加パッケージのインストール
```bash
pip install flash-attn --no-build-isolation
pip install --upgrade accelerate
pip install datasets
```

## 環境管理

### 環境を有効化
```bash
conda activate sft
```

### 環境を終了
```bash
conda deactivate
```

---

## モデルの準備

`model` フォルダの中に、学習させたいモデルを配置してください。

---

## トレーニング実行

### 1. 通常のトレーニング
```bash
python src/train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --max_seq_length 4096 \
    --data_files ./data/sample_data.jsonl \
    --model_name_or_path ./model/sample_model \
    --output_dir results/
```

### 2. chat_templateを使ったチャットトレーニング
`src/train_chat.py` では以下の `chat_template` が使用されています。

```python
chat_template = (
    "{{bos_token}}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{ '\n\n### 指示:\n' + message['content'] }}"
    "{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}"
    "{% elif message['role'] == 'assistant' %}{{ '\n\n### 応答:\n' + message['content'] + eos_token }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}{{ '\n\n### 応答:\n' }}{% endif %}"
    "{% endfor %}"
)
```

実行コマンド例:
```bash
python src/train_chat.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --data_files ./data/sample_data.jsonl \
    --model_name_or_path ./model/sample_model \
    --output_dir results/output_model/ \
    --wandb_project "sample-sft" \
    --wandb_run_name "test_1" \
    --wandb_log_steps 10
```

---


