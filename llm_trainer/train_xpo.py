#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from trl.experimental.xpo import XPOConfig, XPOTrainer
from trl.trainer.judges import BasePairwiseJudge

console = Console()

DEFAULT_MODEL = "google/gemma-3-270m-it"
DEFAULT_DATASET = "trl-lib/ultrafeedback-prompt"
DEFAULT_RERANKER = "Qwen/Qwen3-Reranker-4B"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )
    logger = logging.getLogger("llm_trainer")
    return logger


class QuantizationType(str, Enum):
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


class DatasetFormat(str, Enum):
    PROMPT = "prompt"
    MESSAGES = "messages"
    INSTRUCTION = "instruction"


class Qwen3RerankerJudge(BasePairwiseJudge):
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_RERANKER,
        device: str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        max_length: int = 8192,
        quantize_4bit: bool = True,
        instruction: str = (
            "Evaluate how human-like the assistant's response is. Consider natural language flow, "
            "coherence, appropriate tone, genuine helpfulness, and absence of robotic or repetitive patterns. "
            "A human-like response should feel conversational, nuanced, and contextually appropriate."
        ),
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.instruction = instruction
        self.torch_dtype = torch_dtype
        self.quantize_4bit = quantize_4bit
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._model = None
        self._tokenizer = None
        self._prefix_tokens = None
        self._suffix_tokens = None
        self._token_true_id = None
        self._token_false_id = None

    def _get_quantization_config(self) -> BitsAndBytesConfig | None:
        if not self.quantize_4bit:
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side="left")
        quant_config = self._get_quantization_config()
        model_kwargs: dict[str, Any] = {
            "device_map": self.device,
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **model_kwargs,
        ).eval()
        self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
        self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

    def _format_input(self, query: str, document: str) -> str:
        return f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"

    def _compute_score(self, query: str, document: str) -> float:
        self._load_model()
        formatted = self._format_input(query, document)
        inputs = self._tokenizer(
            formatted,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=self.max_length - len(self._prefix_tokens) - len(self._suffix_tokens),
        )
        input_ids = self._prefix_tokens + inputs["input_ids"] + self._suffix_tokens
        input_ids = torch.tensor([input_ids], device=self._model.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            true_logit = logits[:, self._token_true_id]
            false_logit = logits[:, self._token_false_id]
            scores = torch.stack([false_logit, true_logit], dim=1)
            probs = torch.nn.functional.softmax(scores, dim=1)
            score = probs[:, 1].item()
        return score

    def _extract_query_from_prompt(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
            if prompt and isinstance(prompt[0], dict):
                return prompt[0].get("content", str(prompt[0]))
            return str(prompt)
        return str(prompt)

    def judge(
        self,
        prompts: list[str],
        completions: list[tuple[str, str]],
        shuffle_order: bool = True,
    ) -> list[int]:
        self._load_model()
        results = []
        for prompt, (completion_a, completion_b) in zip(prompts, completions, strict=True):
            query = self._extract_query_from_prompt(prompt)
            score_a = self._compute_score(query, completion_a)
            score_b = self._compute_score(query, completion_b)
            winner = 0 if score_a >= score_b else 1
            results.append(winner)
        return results


@dataclass
class ModelConfig:
    model_name_or_path: str = DEFAULT_MODEL
    tokenizer_name_or_path: str | None = None
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    trust_remote_code: bool = False
    use_flash_attention: bool = False
    quantization: QuantizationType = QuantizationType.NONE
    device_map: str | dict[str, Any] = "auto"
    attn_implementation: str | None = None
    max_memory: dict[int, str] | None = None

    def __post_init__(self) -> None:
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path
        if self.use_flash_attention and self.attn_implementation is None:
            self.attn_implementation = "flash_attention_2"

    def get_torch_dtype(self) -> torch.dtype | str:
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map[self.torch_dtype]

    def get_quantization_config(self) -> BitsAndBytesConfig | None:
        if self.quantization == QuantizationType.NONE:
            return None
        if self.quantization == QuantizationType.INT8:
            return BitsAndBytesConfig(load_in_8bit=True)
        if self.quantization == QuantizationType.INT4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="fp4",
            )
        if self.quantization == QuantizationType.NF4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        return None


@dataclass
class PeftConfig:
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | str = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: list[str] | None = None
    use_rslora: bool = False
    use_dora: bool = False

    def to_lora_config(self) -> LoraConfig:
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
            "SEQ_CLS": TaskType.SEQ_CLS,
        }
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules if isinstance(self.target_modules, list) else [self.target_modules],
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            modules_to_save=self.modules_to_save,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )


@dataclass
class DataConfig:
    dataset_name_or_path: str = DEFAULT_DATASET
    dataset_config: str | None = None
    dataset_split: str = "train"
    validation_split: str | None = None
    validation_split_percentage: float = 0.1
    format: DatasetFormat = DatasetFormat.PROMPT
    messages_column: str = "messages"
    instruction_column: str = "instruction"
    input_column: str = "input"
    output_column: str = "output"
    prompt_column: str = "prompt"
    max_length: int | None = None
    num_proc: int = 4
    streaming: bool = False
    shuffle: bool = True
    shuffle_seed: int = 42


@dataclass
class JudgeConfig:
    model_name_or_path: str = DEFAULT_RERANKER
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    max_length: int = 8192
    quantize_4bit: bool = True
    instruction: str = "Given a user query, evaluate if the assistant response adequately answers the query"


@dataclass
class XPOTrainingConfig:
    max_new_tokens: int = 64
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int | None = None
    beta: list[float] = field(default_factory=lambda: [0.1])
    alpha: list[float] = field(default_factory=lambda: [1e-5])
    loss_type: Literal["sigmoid", "ipo"] = "sigmoid"
    disable_dropout: bool = True
    missing_eos_penalty: float | None = None


@dataclass
class TrainingConfig:
    output_dir: str = "./output"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-7
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_torch"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict[str, Any] | None = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    seed: int = 42
    report_to: list[str] | str = field(default_factory=lambda: ["none"])
    run_name: str | None = None
    hub_model_id: str | None = None
    push_to_hub: bool = False
    hub_private_repo: bool = True
    resume_from_checkpoint: str | bool | None = None
    max_steps: int = -1
    ddp_find_unused_parameters: bool = False
    xpo: XPOTrainingConfig = field(default_factory=XPOTrainingConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def __post_init__(self) -> None:
        if self.gradient_checkpointing and self.gradient_checkpointing_kwargs is None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": False}

    def to_xpo_config(self, data_config: DataConfig, max_length: int) -> XPOConfig:
        return XPOConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            warmup_steps=self.warmup_steps,
            max_grad_norm=self.max_grad_norm,
            lr_scheduler_type=self.lr_scheduler_type,
            optim=self.optim,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            fp16=self.fp16,
            bf16=self.bf16,
            tf32=self.tf32,
            gradient_checkpointing=self.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.gradient_checkpointing_kwargs,
            dataloader_num_workers=self.dataloader_num_workers,
            dataloader_pin_memory=self.dataloader_pin_memory,
            seed=self.seed,
            report_to=self.report_to if isinstance(self.report_to, list) else [self.report_to],
            run_name=self.run_name,
            hub_model_id=self.hub_model_id,
            push_to_hub=self.push_to_hub,
            hub_private_repo=self.hub_private_repo,
            max_steps=self.max_steps,
            ddp_find_unused_parameters=self.ddp_find_unused_parameters,
            max_length=max_length,
            max_new_tokens=self.xpo.max_new_tokens,
            temperature=self.xpo.temperature,
            top_p=self.xpo.top_p,
            top_k=self.xpo.top_k,
            beta=self.xpo.beta,
            alpha=self.xpo.alpha,
            loss_type=self.xpo.loss_type,
            disable_dropout=self.xpo.disable_dropout,
            missing_eos_penalty=self.xpo.missing_eos_penalty,
        )


@dataclass
class FullConfig:
    model: ModelConfig
    peft: PeftConfig
    data: DataConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> FullConfig:
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> FullConfig:
        model_dict = config_dict.get("model", {})
        if "quantization" in model_dict and isinstance(model_dict["quantization"], str):
            model_dict["quantization"] = QuantizationType(model_dict["quantization"])
        data_dict = config_dict.get("data", {})
        if "format" in data_dict and isinstance(data_dict["format"], str):
            data_dict["format"] = DatasetFormat(data_dict["format"])
        training_dict = config_dict.get("training", {})
        if "xpo" in training_dict and isinstance(training_dict["xpo"], dict):
            training_dict["xpo"] = XPOTrainingConfig(**training_dict["xpo"])
        if "judge" in training_dict and isinstance(training_dict["judge"], dict):
            training_dict["judge"] = JudgeConfig(**training_dict["judge"])
        model_config = ModelConfig(**model_dict)
        peft_config = PeftConfig(**config_dict.get("peft", {}))
        data_config = DataConfig(**data_dict)
        training_config = TrainingConfig(**training_dict)
        return cls(model=model_config, peft=peft_config, data=data_config, training=training_config)

    def to_dict(self) -> dict[str, Any]:
        def dataclass_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for k in obj.__dataclass_fields__:
                    v = getattr(obj, k)
                    result[k] = dataclass_to_dict(v)
                return result
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            if isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            return obj

        return {
            "model": dataclass_to_dict(self.model),
            "peft": dataclass_to_dict(self.peft),
            "data": dataclass_to_dict(self.data),
            "training": dataclass_to_dict(self.training),
        }

    def save_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def get_model_max_length(model_name_or_path: str, trust_remote_code: bool = False) -> int:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    max_length = getattr(config, "max_position_embeddings", None)
    if max_length is None:
        max_length = getattr(config, "n_positions", None)
    if max_length is None:
        max_length = getattr(config, "max_sequence_length", None)
    if max_length is None:
        max_length = getattr(config, "seq_length", None)
    if max_length is None:
        max_length = 2048
    return max_length


def load_tokenizer(config: ModelConfig, logger: logging.Logger) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer from: {config.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    return tokenizer


def load_model(config: ModelConfig, logger: logging.Logger) -> PreTrainedModel:
    logger.info(f"Loading model from: {config.model_name_or_path}")
    quant_config = config.get_quantization_config()
    if quant_config is not None:
        logger.info(f"Using quantization: {config.quantization.value}")
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": config.model_name_or_path,
        "dtype": config.get_torch_dtype(),
        "trust_remote_code": config.trust_remote_code,
        "device_map": config.device_map,
        "quantization_config": quant_config,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    if config.max_memory:
        model_kwargs["max_memory"] = config.max_memory
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    logger.info(f"Model loaded. Parameters: {model.num_parameters():,}")
    return model


def apply_peft(model: PreTrainedModel, config: PeftConfig, logger: logging.Logger) -> PreTrainedModel:
    if not config.enabled:
        logger.info("PEFT disabled, using full model fine-tuning")
        return model
    logger.info("Applying PEFT/LoRA configuration")
    lora_config = config.to_lora_config()
    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    trainable_percent = 100 * trainable_params / all_params
    logger.info(f"Trainable parameters: {trainable_params:,} / {all_params:,} ({trainable_percent:.2f}%)")
    return model


def load_and_prepare_dataset(
    config: DataConfig,
    logger: logging.Logger,
) -> tuple[Dataset, Dataset | None]:
    logger.info(f"Loading dataset: {config.dataset_name_or_path}")
    if Path(config.dataset_name_or_path).exists():
        if config.dataset_name_or_path.endswith(".json") or config.dataset_name_or_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=config.dataset_name_or_path, split="train")
        elif config.dataset_name_or_path.endswith(".csv"):
            dataset = load_dataset("csv", data_files=config.dataset_name_or_path, split="train")
        elif config.dataset_name_or_path.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=config.dataset_name_or_path, split="train")
        else:
            dataset = load_dataset(config.dataset_name_or_path, split=config.dataset_split)
    else:
        load_kwargs: dict[str, Any] = {
            "path": config.dataset_name_or_path,
            "split": config.dataset_split,
            "streaming": config.streaming,
        }
        if config.dataset_config:
            load_kwargs["name"] = config.dataset_config
        dataset = load_dataset(**load_kwargs)
    logger.info(f"Dataset loaded. Size: {len(dataset) if hasattr(dataset, '__len__') else 'streaming'}")
    if config.format == DatasetFormat.PROMPT:
        if config.prompt_column != "prompt" and config.prompt_column in dataset.column_names:
            dataset = dataset.rename_column(config.prompt_column, "prompt")
    elif config.format == DatasetFormat.MESSAGES:
        logger.info("Converting messages format to prompt format for XPO...")

        def extract_prompt(examples: dict[str, Any]) -> dict[str, list[Any]]:
            prompts = []
            for messages in examples[config.messages_column]:
                if isinstance(messages, str):
                    messages = json.loads(messages)
                user_msgs = [m for m in messages if m.get("role") == "user"]
                prompts.append(user_msgs if user_msgs else messages[:-1] if messages else [])
            return {"prompt": prompts}

        dataset = dataset.map(extract_prompt, batched=True, num_proc=config.num_proc)
    elif config.format == DatasetFormat.INSTRUCTION:
        logger.info("Converting instruction format to prompt format for XPO...")

        def extract_instruction_prompt(examples: dict[str, Any]) -> dict[str, list[Any]]:
            prompts = []
            instructions = examples[config.instruction_column]
            inputs = examples.get(config.input_column, [""] * len(instructions))
            for instruction, input_text in zip(instructions, inputs, strict=True):
                content = f"{instruction}\n\n{input_text}" if input_text else instruction
                prompts.append([{"role": "user", "content": content}])
            return {"prompt": prompts}

        dataset = dataset.map(extract_instruction_prompt, batched=True, num_proc=config.num_proc)
    if config.shuffle and not config.streaming:
        dataset = dataset.shuffle(seed=config.shuffle_seed)
    eval_dataset = None
    if config.validation_split_percentage > 0 and not config.streaming:
        split_dataset = dataset.train_test_split(
            test_size=config.validation_split_percentage, seed=config.shuffle_seed
        )
        dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        logger.info(f"Split dataset - Train: {len(dataset)}, Eval: {len(eval_dataset)}")
    return dataset, eval_dataset


def display_config_summary(config: FullConfig, max_length: int, logger: logging.Logger) -> None:
    table = Table(title="XPO Training Configuration Summary", show_header=True, header_style="bold magenta")
    table.add_column("Category", style="cyan", width=15)
    table.add_column("Parameter", style="green")
    table.add_column("Value", style="yellow")
    table.add_row("Model", "Name", config.model.model_name_or_path)
    table.add_row("", "Dtype", config.model.torch_dtype)
    table.add_row("", "Quantization", config.model.quantization.value)
    table.add_row("", "Flash Attention", str(config.model.use_flash_attention))
    table.add_row("PEFT", "Enabled", str(config.peft.enabled))
    if config.peft.enabled:
        table.add_row("", "LoRA Rank (r)", str(config.peft.r))
        table.add_row("", "LoRA Alpha", str(config.peft.lora_alpha))
        table.add_row("", "LoRA Dropout", str(config.peft.lora_dropout))
        table.add_row("", "Target Modules", str(config.peft.target_modules))
    table.add_row("Data", "Dataset", config.data.dataset_name_or_path)
    table.add_row("", "Format", config.data.format.value)
    max_length_display = f"{max_length} (auto)" if config.data.max_length is None else str(max_length)
    table.add_row("", "Max Length", max_length_display)
    table.add_row("Training", "Output Dir", config.training.output_dir)
    table.add_row("", "Epochs", str(config.training.num_train_epochs))
    table.add_row("", "Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("", "Grad Accum Steps", str(config.training.gradient_accumulation_steps))
    table.add_row("", "Learning Rate", str(config.training.learning_rate))
    table.add_row("", "Scheduler", config.training.lr_scheduler_type)
    table.add_row("XPO", "Max New Tokens", str(config.training.xpo.max_new_tokens))
    table.add_row("", "Temperature", str(config.training.xpo.temperature))
    table.add_row("", "Beta", str(config.training.xpo.beta))
    table.add_row("", "Alpha", str(config.training.xpo.alpha))
    table.add_row("", "Loss Type", config.training.xpo.loss_type)
    table.add_row("Judge", "Model", config.training.judge.model_name_or_path)
    table.add_row("", "Max Length", str(config.training.judge.max_length))
    console.print(table)


def run_xpo_training(config: FullConfig, logger: logging.Logger | None = None) -> XPOTrainer:
    if logger is None:
        logger = setup_logging()
    console.print(
        Panel.fit("[bold green]LLM Fine-Tuning with XPO, PEFT, and TRL[/bold green]", border_style="green")
    )
    logger.info(f"Fetching model config to determine max_length from: {config.model.model_name_or_path}")
    model_max_length = get_model_max_length(config.model.model_name_or_path, config.model.trust_remote_code)
    if config.data.max_length is None:
        max_length = model_max_length
        logger.info(f"Auto-detected max_length from model config: {max_length}")
    else:
        max_length = config.data.max_length
        if max_length > model_max_length:
            logger.warning(
                f"Configured max_length ({max_length}) exceeds model's max_position_embeddings ({model_max_length}). "
                f"Using model's limit: {model_max_length}"
            )
            max_length = model_max_length
    display_config_summary(config, max_length, logger)
    os.makedirs(config.training.output_dir, exist_ok=True)
    config.save_yaml(Path(config.training.output_dir) / "training_config.yaml")
    tokenizer = load_tokenizer(config.model, logger)
    model = load_model(config.model, logger)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logger.info("Resizing model embeddings to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))
    model = apply_peft(model, config.peft, logger)
    train_dataset, eval_dataset = load_and_prepare_dataset(config.data, logger)
    logger.info(f"Initializing Qwen3 Reranker Judge: {config.training.judge.model_name_or_path}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    judge = Qwen3RerankerJudge(
        model_name_or_path=config.training.judge.model_name_or_path,
        torch_dtype=dtype_map[config.training.judge.torch_dtype],
        max_length=config.training.judge.max_length,
        quantize_4bit=config.training.judge.quantize_4bit,
        instruction=config.training.judge.instruction,
    )
    xpo_config = config.training.to_xpo_config(config.data, max_length)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": xpo_config,
        "train_dataset": train_dataset,
        "processing_class": tokenizer,
        "judge": judge,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = XPOTrainer(**trainer_kwargs)
    logger.info("Starting XPO training...")
    train_result = trainer.train(resume_from_checkpoint=config.training.resume_from_checkpoint)
    logger.info("Training completed!")
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if config.peft.enabled:
        logger.info("Merging LoRA adapter with base model...")
        merged_model = trainer.model.merge_and_unload()
        merged_output_dir = Path(config.training.output_dir) / "merged"
        merged_output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
        logger.info(f"Merged model saved to: {merged_output_dir}")
        console.print(f"[green]Merged model saved to: {merged_output_dir}[/green]")
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    if config.training.push_to_hub:
        logger.info("Pushing model to Hub...")
        trainer.push_to_hub()
    console.print(Panel.fit("[bold green]XPO Training Complete![/bold green]", border_style="green"))
    return trainer


def run_training(config: FullConfig, logger: logging.Logger | None = None) -> XPOTrainer:
    return run_xpo_training(config, logger)


def generate_example_config(output_path: str = "config.yaml") -> None:
    config = FullConfig(
        model=ModelConfig(
            model_name_or_path=DEFAULT_MODEL,
            torch_dtype="bfloat16",
            quantization=QuantizationType.NF4,
        ),
        peft=PeftConfig(
            enabled=True,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        data=DataConfig(
            dataset_name_or_path=DEFAULT_DATASET,
            format=DatasetFormat.PROMPT,
            prompt_column="prompt",
            max_length=None,
            validation_split_percentage=0.05,
        ),
        training=TrainingConfig(
            output_dir="./outputs/xpo-training",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=5e-7,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=True,
            gradient_checkpointing=True,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            report_to=["none"],
            xpo=XPOTrainingConfig(
                max_new_tokens=64,
                temperature=0.9,
                beta=[0.1],
                alpha=[1e-5],
                loss_type="sigmoid",
            ),
            judge=JudgeConfig(
                model_name_or_path=DEFAULT_RERANKER,
                torch_dtype="bfloat16",
                max_length=8192,
                quantize_4bit=True,
            ),
        ),
    )
    config.save_yaml(output_path)
    console.print(f"[green]Example XPO config saved to: {output_path}[/green]")


def main() -> None:
    import typer

    app = typer.Typer(help="LLM Fine-Tuning with XPO, PEFT, and TRL", no_args_is_help=True)

    @app.command()
    def train(
        config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
        log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    ) -> None:
        logger = setup_logging(log_level)
        try:
            config = FullConfig.from_yaml(config_path)
            run_xpo_training(config, logger)
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise typer.Exit(1) from None
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file: {e}")
            raise typer.Exit(1) from None
        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise typer.Exit(1) from None

    @app.command()
    def generate_config(
        output_path: str = typer.Option("config.yaml", "--output", "-o", help="Output path for config file"),
    ) -> None:
        generate_example_config(output_path)

    @app.command()
    def validate_config(
        config_path: str = typer.Argument(..., help="Path to YAML configuration file"),
    ) -> None:
        logger = setup_logging()
        try:
            config = FullConfig.from_yaml(config_path)
            model_max_length = get_model_max_length(config.model.model_name_or_path, config.model.trust_remote_code)
            max_length = config.data.max_length if config.data.max_length is not None else model_max_length
            console.print("[green]Configuration is valid![/green]")
            display_config_summary(config, max_length, logger)
        except Exception as e:
            console.print(f"[red]Configuration validation failed: {e}[/red]")
            raise typer.Exit(1) from None

    app()


if __name__ == "__main__":
    main()
