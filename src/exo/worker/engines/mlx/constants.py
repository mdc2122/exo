import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


KV_GROUP_SIZE: int | None = 32
KV_BITS: int | None = None
ATTENTION_KV_BITS: int | None = 4
MAX_TOKENS: int = 32168
MAX_KV_SIZE: int | None = 3200
KEEP_KV_SIZE: int | None = 1600
QUANTIZE_MODEL_MODE: str | None = "affine"
CACHE_GROUP_SIZE: int = 64
KV_CACHE_BITS: int | None = None
TURBOQUANT_KV_BITS: int | None = _int_env("EXO_TURBOQUANT_KV_BITS")
TURBOQUANT_KV_SEED: int = _int_env("EXO_TURBOQUANT_KV_SEED") or 42
TURBOQUANT_FUSED: bool = (
    os.environ.get("EXO_TURBOQUANT_FUSED", "false").lower() == "true"
)

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
