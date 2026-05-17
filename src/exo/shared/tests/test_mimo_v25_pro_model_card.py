import pytest
from anyio import Path
from pydantic import ValidationError

from exo.shared.models.model_cards import ConfigData, ModelCard, ModelTask


async def test_mimo_v25_pro_builtin_model_card_is_text_only_agentic_long_context() -> (
    None
):
    card = await ModelCard.load_from_path(
        Path("resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml")
    )

    assert card.model_id == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert card.tasks == [ModelTask.TextGeneration]
    assert card.capabilities == [
        "text",
        "thinking",
        "thinking_toggle",
        "agentic",
        "long-context",
        "code",
    ]
    assert card.context_length == 1_048_576
    assert card.hidden_size == 6_144
    assert card.n_layers == 70
    assert card.num_key_value_heads == 8
    assert card.architecture == "MiMoV2ForCausalLM"
    assert card.supports_tensor is True
    assert card.vision is None

    media_markers = {
        "image",
        "images",
        "speech",
        "vision",
        "video",
        "audio",
        "omni",
        "omnimodal",
        "multimodal",
    }
    assert media_markers.isdisjoint(
        {capability.lower() for capability in card.capabilities}
    )


@pytest.mark.parametrize(
    "media_capability",
    [
        "vision",
        "image",
        "video",
        "audio",
        "speech",
        "multimodal",
        "omnimodal",
    ],
)
def test_mimo_v25_pro_model_card_rejects_media_capability_metadata(
    media_capability: str,
) -> None:
    with pytest.raises(ValidationError, match="must not declare media capabilities"):
        ModelCard.model_validate(
            {
                "model_id": "XiaomiMiMo/MiMo-V2.5-Pro",
                "storage_size": {"in_bytes": 1},
                "n_layers": 70,
                "hidden_size": 6144,
                "num_key_value_heads": 8,
                "supports_tensor": True,
                "tasks": ["TextGeneration"],
                "architecture": "MiMoV2ForCausalLM",
                "capabilities": ["text", media_capability],
                "context_length": 1_048_576,
            }
        )


@pytest.mark.parametrize(
    "media_task",
    [
        "TextToImage",
        "ImageToImage",
    ],
)
def test_mimo_v25_pro_model_card_rejects_non_text_generation_tasks(
    media_task: str,
) -> None:
    with pytest.raises(ValidationError, match="supports text generation only"):
        ModelCard.model_validate(
            {
                "model_id": "XiaomiMiMo/MiMo-V2.5-Pro",
                "storage_size": {"in_bytes": 1},
                "n_layers": 70,
                "hidden_size": 6144,
                "num_key_value_heads": 8,
                "supports_tensor": True,
                "tasks": ["TextGeneration", media_task],
                "architecture": "MiMoV2ForCausalLM",
                "capabilities": ["text", "agentic", "long-context"],
                "context_length": 1_048_576,
            }
        )


def test_mimo_v25_pro_config_data_allows_text_causal_lm_when_mtp_absent() -> None:
    config_data = ConfigData.model_validate(
        {
            "architectures": ["MiMoV2ForCausalLM"],
            "num_hidden_layers": 70,
            "hidden_size": 6144,
            "num_key_value_heads": 8,
            "max_position_embeddings": 1_048_576,
        }
    )

    assert config_data.supports_tensor is True


def test_mimo_v25_pro_config_data_allows_text_causal_lm_when_mtp_disabled() -> None:
    config_data = ConfigData.model_validate(
        {
            "architectures": ["MiMoV2ForCausalLM"],
            "num_hidden_layers": 70,
            "hidden_size": 6144,
            "num_key_value_heads": 8,
            "max_position_embeddings": 1_048_576,
            "num_nextn_predict_layers": 0,
            "mtp_enabled": False,
            "use_mtp": False,
            "mtp_config": {"enabled": False, "num_layers": 0},
        }
    )

    assert config_data.supports_tensor is True
    assert config_data.vision is None


@pytest.mark.parametrize(
    "enabled_mtp_metadata",
    [
        {"num_nextn_predict_layers": 1},
        {"mtp_enabled": True},
        {"use_mtp": True},
        {"mtp_config": {"enabled": True}},
        {"mtp_config": {"num_layers": 1}},
    ],
)
def test_mimo_v25_pro_config_data_flags_enabled_mtp_as_not_tensor_supported(
    enabled_mtp_metadata: dict[str, object],
) -> None:
    config_data = ConfigData.model_validate(
        {
            "architectures": ["MiMoV2ForCausalLM"],
            "num_hidden_layers": 70,
            "hidden_size": 6144,
            "num_key_value_heads": 8,
            "max_position_embeddings": 1_048_576,
            **enabled_mtp_metadata,
        }
    )

    assert config_data.supports_tensor is False
    assert config_data.vision is None
