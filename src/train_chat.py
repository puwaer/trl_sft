import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import wandb  # W&B をインポート
from peft import LoraConfig
from datasets import disable_caching, load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import SFTTrainer

disable_caching()

logger = logging.getLogger(__name__)

@dataclass
class SFTTrainingArguments:
    model_name_or_path: str
    data_files: List[str]
    eval_data_files: Optional[List[str]] = None
    tokenizer_name_or_path: Optional[str] = None
    use_fast: bool = True
    additional_special_tokens: Optional[List[str]] = None
    max_seq_length: int = 4096
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_flash_attention_2: bool = False
    use_peft: bool = False
    peft_target_model: Optional[str] = "llm-jp"
    peft_target_modules: Optional[List[str]] = None
    peft_lora_r: int = 8
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    # W&B 関連の引数を追加
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_steps: Optional[int] = None

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("load_in_8bit and load_in_4bit are mutually exclusive")
        if self.peft_target_model and self.peft_target_modules is None:
            if self.peft_target_model == "llm-jp":
                self.peft_target_modules = ["c_attn", "c_proj", "c_fc"]
            elif self.peft_target_model == "llama":
                self.peft_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
                ]
            elif self.peft_target_model == "llama-all":
                self.peft_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "embed_tokens"
                ]
            else:
                logger.warning(
                    f"peft_target_model '{self.peft_target_model}' is not supported, "
                    f"so peft_target_modules is set to None."
                )

    def from_pretrained_kwargs(self, training_args):
        if self.load_in_8bit:
            kwargs = {"load_in_8bit": True}
        elif self.load_in_4bit:
            kwargs = {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            }
        elif training_args.bf16:
            kwargs = {"torch_dtype": torch.bfloat16}
        else:
            kwargs = {"torch_dtype": torch.float16}
        kwargs["use_flash_attention_2"] = self.use_flash_attention_2
        return kwargs

def load_datasets(data_files, tokenizer, max_seq_length=2048):
    datasets = []
    for data_file in data_files:
        dataset = load_dataset("json", data_files=data_file)
        dataset = dataset["train"]

        def tokenize_function(example):
            tokenized = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
                return_dict=True
            )
            return {
                "input_ids": tokenized["input_ids"].squeeze(0).tolist(),
                "attention_mask": tokenized["attention_mask"].squeeze(0).tolist()
            }

        dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
        datasets.append(dataset)
    return concatenate_datasets(datasets)

def main():
    parser = HfArgumentParser((TrainingArguments, SFTTrainingArguments))
    training_args, sft_training_args = parser.parse_args_into_dataclasses()

    # W&B の設定を適用
    if sft_training_args.wandb_project:
        training_args.report_to = ["wandb"]  # W&B にログを送信
        wandb.init(
            project=sft_training_args.wandb_project,
            name=sft_training_args.wandb_run_name,
            config=vars(training_args),  # TrainingArguments を W&B に記録
        )
        if sft_training_args.wandb_log_steps:
            training_args.logging_steps = sft_training_args.wandb_log_steps

    tokenizer_name_or_path = (
        sft_training_args.tokenizer_name_or_path or sft_training_args.model_name_or_path
    )
    logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_fast=sft_training_args.use_fast,
        additional_special_tokens=sft_training_args.additional_special_tokens,
        trust_remote_code=True,
    )

    chat_template = (
        "{{bos_token}}{% for message in messages %}"
        "{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}"
        "{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}"
        "{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}"
        "{% endfor %}"
    )
    tokenizer.chat_template = chat_template
    logger.info("Custom chat template applied to tokenizer")

    if tokenizer.bos_token is None:
        tokenizer.bos_token = "<|begin_of_text|>"
        logger.info(f"Set default bos_token: {tokenizer.bos_token}")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|end_of_text|>"
        logger.info(f"Set default eos_token: {tokenizer.eos_token}")

    logger.info("Loading data")
    train_dataset = load_datasets(sft_training_args.data_files, tokenizer, sft_training_args.max_seq_length)
    if sft_training_args.eval_data_files:
        eval_dataset = load_datasets(sft_training_args.eval_data_files, tokenizer, sft_training_args.max_seq_length)
        training_args.do_eval = True
    else:
        eval_dataset = None

    logger.info(f"Loading model from {sft_training_args.model_name_or_path}")
    kwargs = sft_training_args.from_pretrained_kwargs(training_args)
    model = AutoModelForCausalLM.from_pretrained(
        sft_training_args.model_name_or_path,
        trust_remote_code=True,
        **kwargs,
    )

    peft_config = None
    if sft_training_args.use_peft:
        logger.info("Setting up LoRA")
        peft_config = LoraConfig(
            r=sft_training_args.peft_lora_r,
            target_modules=sft_training_args.peft_target_modules,
            lora_alpha=sft_training_args.peft_lora_alpha,
            lora_dropout=sft_training_args.peft_lora_dropout,
            fan_in_fan_out=True,
            bias="none",
            task_type="CAUSAL_LM",
        )
        if training_args.gradient_checkpointing:
            for param in model.parameters():
                param.requires_grad = False
                if param.ndim == 1:
                    param.data = param.data.to(torch.float32)
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

    logger.info("Setting up trainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    logger.info("Training")
    trainer.train()

    logger.info("Saving model")
    trainer.save_model()

    # W&B を終了
    if sft_training_args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s:%(lineno)d: %(levelname)s: %(message)s",
    )
    main()