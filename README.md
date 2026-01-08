# LLM Trainer

Uses TRL and PEFT. Supports any Qwen3 Reranker as a judge.

Default configuration (`config.yaml`) requires 19GB of VRAM.

### Run
```
uv sync

uv run xpo train config.yaml
```

### Infer
```
uv run infer generate "Hello! How is it going?" -m ./outputs/xpo-training/merged
```