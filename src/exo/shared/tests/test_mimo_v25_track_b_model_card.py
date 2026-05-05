from anyio import Path

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId


def _media_markers(card: ModelCard) -> set[str]:
    return {capability.lower() for capability in card.capabilities}


async def test_mimo_v25_track_b_builtin_model_card_is_multimodal_fp8() -> None:
    card = await ModelCard.load_from_path(
        Path("resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5.toml")
    )

    assert card.model_id == "XiaomiMiMo/MiMo-V2.5"
    assert card.quantization == "fp8"
    assert card.tasks == [ModelTask.TextGeneration]
    assert card.capabilities == [
        "text",
        "image",
        "video",
        "audio",
        "agentic",
        "long-context",
    ]
    assert card.context_length == 1_048_576
    assert card.hidden_size == 4_096
    assert card.n_layers == 48
    assert card.num_key_value_heads == 4
    assert card.architecture == "MiMoV2ForCausalLM"
    assert card.supports_tensor is True
    assert card.storage_size.in_bytes == 315_031_102_208

    assert card.vision is not None
    assert card.vision.model_type == "mimovl"
    assert card.vision.image_token_id == 151_655
    assert card.vision.image_token == "<|image_pad|>"
    assert card.vision.weights_repo == "XiaomiMiMo/MiMo-V2.5"
    assert card.vision.processor_repo == "XiaomiMiMo/MiMo-V2.5"
    assert card.vision.sample_fps == 2.0
    assert card.vision.temporal_merge_kernel_size == 2

    assert {"image", "video", "audio"}.issubset(_media_markers(card))


async def test_mimo_v25_track_b_builtin_card_loads_separately_from_pro() -> None:
    pro_card = await ModelCard.load(ModelId("XiaomiMiMo/MiMo-V2.5-Pro"))
    track_b_card = await ModelCard.load(ModelId("XiaomiMiMo/MiMo-V2.5"))

    assert pro_card.model_id == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert pro_card.vision is None
    assert "video" not in {capability.lower() for capability in pro_card.capabilities}

    assert track_b_card.model_id == "XiaomiMiMo/MiMo-V2.5"
    assert track_b_card.vision is not None
    assert "video" in {capability.lower() for capability in track_b_card.capabilities}
