# Copyright © 2023 Apple Inc.

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_unflatten

from . import whisper


def _convert_hf_config(config: dict) -> dict:
    """Convert HuggingFace Whisper config keys to MLX format."""
    d_model = config["d_model"]
    return {
        "n_mels": config["num_mel_bins"],
        "n_audio_ctx": config["max_source_positions"] // 2,
        "n_audio_state": d_model,
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_vocab": config["vocab_size"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": d_model,
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }


def _convert_hf_weights(weights: dict) -> dict:
    """Convert HuggingFace Whisper weight keys to MLX format."""
    key_map = {
        "model.encoder.layers.": "encoder.blocks.",
        "model.decoder.layers.": "decoder.blocks.",
        "model.encoder.layer_norm.": "encoder.ln_post.",
        "model.decoder.layer_norm.": "decoder.ln.",
        "model.decoder.embed_tokens.": "decoder.token_embedding.",
        "model.encoder.conv1.": "encoder.conv1.",
        "model.encoder.conv2.": "encoder.conv2.",
        "self_attn.q_proj.": "attn.query.",
        "self_attn.k_proj.": "attn.key.",
        "self_attn.v_proj.": "attn.value.",
        "self_attn.out_proj.": "attn.out.",
        "encoder_attn.q_proj.": "cross_attn.query.",
        "encoder_attn.k_proj.": "cross_attn.key.",
        "encoder_attn.v_proj.": "cross_attn.value.",
        "encoder_attn.out_proj.": "cross_attn.out.",
        "self_attn_layer_norm.": "attn_ln.",
        "encoder_attn_layer_norm.": "cross_attn_ln.",
        "final_layer_norm.": "mlp_ln.",
        "fc1.": "mlp1.",
        "fc2.": "mlp2.",
    }

    converted = {}
    for k, v in weights.items():
        new_key = k
        for hf, mlx in key_map.items():
            new_key = new_key.replace(hf, mlx)

        # Positional embeddings: drop .weight suffix, flatten into parent
        if new_key == "model.encoder.embed_positions.weight":
            new_key = "encoder._positional_embedding"
        elif new_key == "model.decoder.embed_positions.weight":
            new_key = "decoder.positional_embedding"

        # Conv weights: HF is (out, in, kernel), MLX expects (out, kernel, in)
        if new_key in ("encoder.conv1.weight", "encoder.conv2.weight"):
            v = mx.transpose(v, axes=(0, 2, 1))

        converted[new_key] = v

    return converted


def load_model(
    path_or_hf_repo: str,
    dtype: mx.Dtype = mx.float32,
) -> whisper.Whisper:
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(snapshot_download(repo_id=path_or_hf_repo))

    with open(str(model_path / "config.json"), "r") as f:
        config = json.loads(f.read())
        config.pop("model_type", None)
        quantization = config.pop("quantization", None)

    # Detect HuggingFace format and convert to MLX format
    is_hf = "d_model" in config
    if is_hf:
        config = _convert_hf_config(config)

    model_args = whisper.ModelDimensions(**config)

    # Prefer model.safetensors, fall back to weights.safetensors, then weights.npz
    wf = model_path / "model.safetensors"
    if not wf.exists():
        wf = model_path / "weights.safetensors"
    if not wf.exists():
        wf = model_path / "weights.npz"
    weights = mx.load(str(wf))

    # Convert HF weight keys to MLX format
    if is_hf:
        weights = _convert_hf_weights(weights)

    weights = {
        k: v.astype(dtype)
        if v.dtype in (mx.float16, mx.bfloat16, mx.float32) and v.dtype != dtype
        else v
        for k, v in weights.items()
    }

    model = whisper.Whisper(model_args, dtype)

    if quantization is not None:
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = tree_unflatten(list(weights.items()))
    model.update(weights)
    mx.eval(model.parameters())
    return model
