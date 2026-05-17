from anyio import Path

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId


async def test_mimo_v25_pro_6bit_builtin_model_card_is_separate_text_only_identity() -> None:
    card = await ModelCard.load_from_path(
        Path("resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro-6bit-MLX.toml")
    )

    assert card.model_id == "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX"
    assert card.base_model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert card.quantization == "6bit-mlx-affine"
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


async def test_mimo_v25_pro_6bit_builtin_card_loads_from_registry_separately() -> None:
    official_card = await ModelCard.load(ModelId("XiaomiMiMo/MiMo-V2.5-Pro"))
    custom_6bit_card = await ModelCard.load(
        ModelId("XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX")
    )

    assert official_card.model_id == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert custom_6bit_card.model_id == "XiaomiMiMo/MiMo-V2.5-Pro-6bit-MLX"
    assert official_card.quantization == "fp8"
    assert custom_6bit_card.quantization == "6bit-mlx-affine"
    assert custom_6bit_card.base_model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert official_card.supports_tensor is True
    assert custom_6bit_card.supports_tensor is True


async def test_kernelpool_mimo_v25_pro_6bit_builtin_card_is_deployed_identity() -> None:
    card = await ModelCard.load_from_path(
        Path("resources/inference_model_cards/kernelpool--MiMo-V2.5-Pro-6bit.toml")
    )

    assert card.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert card.base_model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert card.quantization == "6bit-mlx-affine"
    assert card.tasks == [ModelTask.TextGeneration]
    assert "thinking" in card.capabilities
    assert "thinking_toggle" in card.capabilities
    assert "vision" not in card.capabilities
    assert card.context_length == 1_048_576
    assert card.hidden_size == 6_144
    assert card.n_layers == 70
    assert card.num_key_value_heads == 8
    assert card.architecture == "MiMoV2ForCausalLM"
    assert card.supports_tensor is True
    assert card.storage_size.in_bytes == 835_299_199_488
    assert card.trust_remote_code is True
    assert card.vision is None


async def test_kernelpool_mimo_v25_pro_6bit_card_loads_from_registry() -> None:
    card = await ModelCard.load(ModelId("kernelpool/MiMo-V2.5-Pro-6bit"))

    assert card.model_id == "kernelpool/MiMo-V2.5-Pro-6bit"
    assert card.quantization == "6bit-mlx-affine"
    assert card.base_model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert card.supports_tensor is True
