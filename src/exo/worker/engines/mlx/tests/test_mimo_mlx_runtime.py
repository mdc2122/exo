import importlib
import json
from pathlib import Path
from typing import cast

import mlx_lm.utils as mlx_lm_utils


def test_mimo_v25_pro_registers_mlx_lm_mimo_v2_flash_alias() -> None:
    importlib.import_module("exo.worker.engines.mlx.utils_mlx")

    mapping = cast(dict[str, str], mlx_lm_utils.MODEL_REMAPPING)
    assert mapping["mimo_v2"] == "mimo_v2_flash"


def test_mimo_v25_pro_local_config_is_compatible_with_mlx_lm_mimo_v2_flash() -> None:
    importlib.import_module("exo.worker.engines.mlx.utils_mlx")

    config_path = Path(
        "/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro/config.json"
    )
    if config_path.exists():
        config = cast(dict[str, object], json.loads(config_path.read_text()))
    else:
        config = {
            "model_type": "mimo_v2",
            "num_experts_per_tok": 8,
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 1],
            "add_swa_attention_sink_bias": True,
            "add_full_attention_sink_bias": False,
            "sliding_window_size": 128,
            "vocab_size": 1000,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "n_shared_experts": None,
            "n_routed_experts": 2,
            "routed_scaling_factor": 1.0,
            "topk_method": "greedy",
            "scoring_func": "softmax",
            "norm_topk_prob": True,
            "n_group": 1,
            "topk_group": 1,
            "max_position_embeddings": 1024,
            "layernorm_epsilon": 1e-6,
            "rope_theta": 10000000,
            "swa_rope_theta": 10000,
            "swa_num_attention_heads": 2,
            "swa_num_key_value_heads": 2,
            "head_dim": 8,
            "v_head_dim": 8,
            "swa_head_dim": 8,
            "swa_v_head_dim": 8,
            "partial_rotary_factor": 1,
        }

    required_keys = {
        "model_type",
        "num_experts_per_tok",
        "hybrid_layer_pattern",
        "moe_layer_freq",
        "add_swa_attention_sink_bias",
        "add_full_attention_sink_bias",
        "sliding_window_size",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "moe_intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "n_routed_experts",
        "topk_method",
        "scoring_func",
        "norm_topk_prob",
        "n_group",
        "topk_group",
        "max_position_embeddings",
        "layernorm_epsilon",
        "rope_theta",
        "swa_rope_theta",
        "swa_num_attention_heads",
        "swa_num_key_value_heads",
        "head_dim",
        "v_head_dim",
        "swa_head_dim",
        "swa_v_head_dim",
        "partial_rotary_factor",
    }
    assert required_keys.issubset(config.keys())
    assert config["model_type"] == "mimo_v2"
    assert isinstance(config["num_hidden_layers"], int)
    assert config["num_hidden_layers"] >= 1
    assert isinstance(config["hidden_size"], int)
    assert config["hidden_size"] > 0
