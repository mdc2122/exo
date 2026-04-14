import os

# TODO: Do we want so many constants?
#  I think we want a lot of these as parameters?


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return int(raw)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "off", "no"}


def _str_env(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


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
TURBOQUANT_CACHE_VARIANT: str = _str_env("EXO_TURBOQUANT_CACHE_VARIANT", "v2")
TURBOQUANT_GROUP_SIZE: int = _int_env("EXO_TURBOQUANT_GROUP_SIZE") or CACHE_GROUP_SIZE
TURBOQUANT_KV_SEED: int = _int_env("EXO_TURBOQUANT_KV_SEED") or 42
TURBOQUANT_USE_QJL: bool = _bool_env("EXO_TURBOQUANT_USE_QJL")
TURBOQUANT_USE_ROTATION: bool = _bool_env("EXO_TURBOQUANT_USE_ROTATION", True)
TURBOQUANT_USE_NORMALIZATION: bool = _bool_env("EXO_TURBOQUANT_USE_NORMALIZATION", True)
TURBOQUANT_V3_OUTLIER_CHANNELS: int = (
    _int_env("EXO_TURBOQUANT_V3_OUTLIER_CHANNELS") or 0
)
TURBOQUANT_V3_OUTLIER_BITS: int | None = _int_env("EXO_TURBOQUANT_V3_OUTLIER_BITS")

DEFAULT_TOP_LOGPROBS: int = 5

# TODO: We should really make this opt-in, but Kimi requires trust_remote_code=True
TRUST_REMOTE_CODE: bool = True
