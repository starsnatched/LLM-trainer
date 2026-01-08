from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextStreamer,
)

console = Console()


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="[%H:%M:%S]",
    )
    return logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    model_path: str = "./outputs/xpo-training/merged"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_cache: bool = True
    stream: bool = True
    system_prompt: str | None = None

    def get_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    def to_generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            do_sample=self.do_sample,
            use_cache=self.use_cache,
        )

    @classmethod
    def from_yaml(cls, path: str) -> InferenceConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str) -> None:
        data = {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "attn_implementation": self.attn_implementation,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
            "stream": self.stream,
            "system_prompt": self.system_prompt,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatSession:
    messages: list[ChatMessage] = field(default_factory=list)
    system_prompt: str | None = None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def get_messages_for_template(self) -> list[dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend([msg.to_dict() for msg in self.messages])
        return messages

    def clear(self) -> None:
        self.messages.clear()

    def to_json(self) -> str:
        return json.dumps(
            {
                "system_prompt": self.system_prompt,
                "messages": [msg.to_dict() for msg in self.messages],
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> ChatSession:
        parsed = json.loads(data)
        session = cls(system_prompt=parsed.get("system_prompt"))
        for msg in parsed.get("messages", []):
            session.add_message(msg["role"], msg["content"])
        return session


class InferenceEngine:
    def __init__(self, config: InferenceConfig, logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or setup_logging()
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None
        self.generation_config: GenerationConfig | None = None
        self._load_model()

    def _load_model(self) -> None:
        self.logger.info(f"Loading model from: {self.config.model_path}")
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.config.get_dtype(),
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **model_kwargs,
        )
        self.model.eval()
        self.generation_config = self.config.to_generation_config()
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded. Parameters: {param_count:,}")

    def generate(
        self,
        prompt: str,
        stream: bool | None = None,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        use_stream = stream if stream is not None else self.config.stream
        gen_config = generation_config or self.generation_config
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        streamer = None
        if use_stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
                streamer=streamer,
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    def chat(
        self,
        session: ChatSession,
        user_message: str,
        stream: bool | None = None,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        session.add_user_message(user_message)
        messages = session.get_messages_for_template()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = self.generate(prompt, stream=stream)
        session.add_assistant_message(response)
        return response

    def single_turn(
        self,
        user_message: str,
        system_prompt: str | None = None,
        stream: bool | None = None,
    ) -> str:
        session = ChatSession(system_prompt=system_prompt or self.config.system_prompt)
        return self.chat(session, user_message, stream=stream)


def run_interactive_chat(engine: InferenceEngine) -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Interactive Chat Mode[/bold cyan]\n"
            "Commands: /clear (reset), /system <prompt> (set system), /save <file>, /load <file>, /quit",
            border_style="cyan",
        )
    )
    session = ChatSession(system_prompt=engine.config.system_prompt)
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        if not user_input.strip():
            continue
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if command == "/quit":
                console.print("[yellow]Goodbye![/yellow]")
                break
            elif command == "/clear":
                session.clear()
                console.print("[yellow]Chat history cleared.[/yellow]")
                continue
            elif command == "/system":
                session.system_prompt = arg if arg else None
                console.print(f"[yellow]System prompt set: {arg or '(cleared)'}[/yellow]")
                continue
            elif command == "/save":
                if arg:
                    Path(arg).write_text(session.to_json())
                    console.print(f"[green]Session saved to: {arg}[/green]")
                else:
                    console.print("[red]Usage: /save <filename>[/red]")
                continue
            elif command == "/load":
                if arg and Path(arg).exists():
                    session = ChatSession.from_json(Path(arg).read_text())
                    console.print(f"[green]Session loaded from: {arg}[/green]")
                else:
                    console.print("[red]Usage: /load <filename> (file must exist)[/red]")
                continue
            elif command == "/history":
                console.print(Panel(session.to_json(), title="Chat History", border_style="blue"))
                continue
            else:
                console.print(f"[red]Unknown command: {command}[/red]")
                continue
        console.print("\n[bold blue]Assistant[/bold blue]")
        engine.chat(session, user_input)


def run_batch_inference(
    engine: InferenceEngine,
    input_file: str,
    output_file: str,
    system_prompt: str | None = None,
) -> None:
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_file}[/red]")
        sys.exit(1)
    lines = input_path.read_text().strip().split("\n")
    results = []
    console.print(f"[cyan]Processing {len(lines)} prompts...[/cyan]")
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        console.print(f"[dim]Processing {i + 1}/{len(lines)}...[/dim]")
        response = engine.single_turn(line.strip(), system_prompt=system_prompt, stream=False)
        results.append({"prompt": line.strip(), "response": response})
    output_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Results saved to: {output_file}[/green]")


def generate_example_config(output_path: str = "inference_config.yaml") -> None:
    config = InferenceConfig()
    config.save_yaml(output_path)
    console.print(f"[green]Example inference config saved to: {output_path}[/green]")


def main() -> None:
    import typer

    app = typer.Typer(help="LLM Inference with Merged Models", no_args_is_help=True)

    @app.command()
    def chat(
        model_path: str = typer.Option(
            "./outputs/xpo-training/merged",
            "--model",
            "-m",
            help="Path to merged model directory",
        ),
        config_path: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to inference config YAML",
        ),
        system_prompt: str | None = typer.Option(
            None,
            "--system",
            "-s",
            help="System prompt for the conversation",
        ),
        dtype: str = typer.Option("bfloat16", "--dtype", "-d", help="Model dtype"),
        log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    ) -> None:
        logger = setup_logging(log_level)
        if config_path:
            config = InferenceConfig.from_yaml(config_path)
        else:
            config = InferenceConfig(model_path=model_path, dtype=dtype)
        if system_prompt:
            config.system_prompt = system_prompt
        engine = InferenceEngine(config, logger)
        run_interactive_chat(engine)

    @app.command()
    def generate(
        prompt: str = typer.Argument(..., help="Prompt text for generation"),
        model_path: str = typer.Option(
            "./outputs/xpo-training/merged",
            "--model",
            "-m",
            help="Path to merged model directory",
        ),
        config_path: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to inference config YAML",
        ),
        system_prompt: str | None = typer.Option(
            None,
            "--system",
            "-s",
            help="System prompt",
        ),
        max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum new tokens"),
        temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
        no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output"),
        log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    ) -> None:
        logger = setup_logging(log_level)
        if config_path:
            config = InferenceConfig.from_yaml(config_path)
        else:
            config = InferenceConfig(model_path=model_path)
        config.max_new_tokens = max_tokens
        config.temperature = temperature
        config.stream = not no_stream
        if system_prompt:
            config.system_prompt = system_prompt
        engine = InferenceEngine(config, logger)
        console.print("\n[bold blue]Response[/bold blue]")
        engine.single_turn(prompt, stream=config.stream)
        console.print()

    @app.command()
    def batch(
        input_file: str = typer.Argument(..., help="Input file with one prompt per line"),
        output_file: str = typer.Argument(..., help="Output JSON file for results"),
        model_path: str = typer.Option(
            "./outputs/xpo-training/merged",
            "--model",
            "-m",
            help="Path to merged model directory",
        ),
        config_path: str | None = typer.Option(
            None,
            "--config",
            "-c",
            help="Path to inference config YAML",
        ),
        system_prompt: str | None = typer.Option(
            None,
            "--system",
            "-s",
            help="System prompt for all prompts",
        ),
        log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    ) -> None:
        logger = setup_logging(log_level)
        if config_path:
            config = InferenceConfig.from_yaml(config_path)
        else:
            config = InferenceConfig(model_path=model_path)
        config.stream = False
        engine = InferenceEngine(config, logger)
        run_batch_inference(engine, input_file, output_file, system_prompt)

    @app.command("generate-config")
    def gen_config(
        output_path: str = typer.Option(
            "inference_config.yaml",
            "--output",
            "-o",
            help="Output path for config file",
        ),
    ) -> None:
        generate_example_config(output_path)

    app()


if __name__ == "__main__":
    main()
