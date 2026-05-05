# MiMo V2.5 Pro Live Integration Runlog

## Scope

Integrate the verified XiaomiMiMo/MiMo-V2.5-Pro Track A text-only scaffolding into the isolated Studio2 live exo branch while preserving the already-working Kimi K2.6 video API/adapter behavior. MiMo Pro remains text-only in this pass; media-bearing MiMo Pro requests must fail closed before inference, download, or media fetch/send.

## Branch and Source Snapshot

- Live integration repo: `/Users/studio2/exo`
- Live integration branch: `con-75-mimo-pro-live-integration-20260504-200136`
- Live starting commit: `9b9e366824d7594d1230de4e73e9e569317f5ca6` (`CON-75 preserve live Kimi video state before MiMo integration`)
- Verified Track A source repo: `/Users/studio2/exo-private-mimo-track-a`
- Verified Track A source branch: `con-75-mimo-v25-pro-track-a-impl`
- Verified Track A source commit: `b6d1a80fd024b7e0e4bd29169bd8ec3fdcd46062` (`CON-75: salvage MiMo Pro text-only model card`)

## Merge Strategy

- Use manual file-level merge from verified Track A rather than broad patch application because the live branch contains additional Kimi video API, video upload, video-store, and adapter changes absent from Track A.
- Bring forward only the MiMo Pro text-only model-card/config-validation scaffolding, fixture card, and focused tests.
- Preserve live Kimi video code in `src/exo/api/adapters/chat_completions.py`, `src/exo/api/main.py`, video upload routes/types, and worker video preprocessing paths.
- Add MiMo Pro media fail-closed checks at the chat adapter boundary before any image/video URL fetching or command dispatch.
- Do not add TurboQuant dependencies, TurboQuant model cards, or TurboQuant runtime requirements for MiMo, Kimi, or GLM.
- Do not load or download MiMo Pro weights during this integration; static model-card parsing and adapter/request validation only.

## Verification Log

Focused static verification completed on 2026-05-04 20:07 local time.

Commands and results:

1. `python3 -m py_compile src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py`
   - Result: failed because system `python3` is too old for repository `match` syntax in `chat_completions.py`.
   - Recovery: reran with repo-managed Python via `uv run python`.
2. `uv run python --version && uv run python -m py_compile src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py`
   - Result: passed; Python `3.13.12`.
3. `uv run pytest src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_video_uploads.py -q`
   - Result: passed, `27 passed in 3.28s`.
4. `uv run pytest src/exo/api/tests/test_openai_responses_api.py -q`
   - Result: passed, `4 passed in 0.16s`.
5. Docs/card smoke:
   - `test -s docs/plans/MIMO_V25_PRO_LIVE_INTEGRATION_RUNLOG.md`
   - `test -s resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml`
   - `! grep -Eiq 'image|video|audio|speech|multimodal|omnimodal|vision' resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml`
   - Result: passed; MiMo card remains text-only.
6. TurboQuant grep:
   - `! rg -i 'turboquant' resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py`
   - Result: passed; no TurboQuant dependency introduced in integration files.
7. `uv run ruff check src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_video_uploads.py src/exo/api/tests/test_openai_responses_api.py`
   - Result: passed, `All checks passed!`.
8. `uv run basedpyright src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py`
   - Result: initially failed on live `ErrorChunk.error_code` typing and test `ModelId` typing.
   - Recovery: added typed optional `error_code` to `ErrorChunk` to match existing chat adapter usage and wrapped Kimi test model with `ModelId(...)`.
9. `uv run basedpyright src/exo/shared/models/model_cards.py src/exo/shared/types/chunks.py src/exo/api/adapters/chat_completions.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py`
   - Result: passed, `0 errors, 0 warnings, 0 notes`.
10. Final focused verification bundle:
    - `uv run python -m py_compile src/exo/shared/models/model_cards.py src/exo/shared/types/chunks.py src/exo/api/adapters/chat_completions.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py && uv run pytest src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_video_uploads.py src/exo/api/tests/test_openai_responses_api.py -q && uv run ruff check src/exo/shared/models/model_cards.py src/exo/shared/types/chunks.py src/exo/api/adapters/chat_completions.py src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/api/tests/test_video_uploads.py src/exo/api/tests/test_openai_responses_api.py && test -s docs/plans/MIMO_V25_PRO_LIVE_INTEGRATION_RUNLOG.md && test -s resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml && ! grep -Eiq 'image|video|audio|speech|multimodal|omnimodal|vision' resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml && ! rg -i 'turboquant' resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml src/exo/shared/tests/test_mimo_v25_pro_model_card.py src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py src/exo/shared/models/model_cards.py src/exo/api/adapters/chat_completions.py src/exo/shared/types/chunks.py`
    - Result: passed; pytest `31 passed in 2.23s`; ruff `All checks passed!`.

Existing Kimi/video tests present in this branch:

- `src/exo/api/tests/test_video_uploads.py` (included in focused verification)

## Final Integration Notes

Changed files intended for commit:

- `docs/plans/MIMO_V25_PRO_LIVE_INTEGRATION_RUNLOG.md`
- `resources/inference_model_cards/XiaomiMiMo--MiMo-V2.5-Pro.toml`
- `src/exo/api/adapters/chat_completions.py`
- `src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py`
- `src/exo/shared/models/model_cards.py`
- `src/exo/shared/tests/test_mimo_v25_pro_model_card.py`
- `src/exo/shared/types/chunks.py`

MiMo online status:

- MiMo Pro weights were not loaded or downloaded in this integration pass.
- Integration status is static/offline only: model-card parsing, config validation, request fail-closed validation, and focused regression tests.

Next safe live smoke step:

1. Start exo without enabling any MiMo weight download.
2. Query `/v1/models` and confirm `XiaomiMiMo/MiMo-V2.5-Pro` appears as a text-generation model with text/agentic/long-context capabilities only.
3. Send a text-only dry chat request only after an operator intentionally provisions compatible MiMo Pro weights/config in the model cache.
4. Before any inference smoke, send a MiMo Pro request containing `image_url`, `video_url`, and an audio-shaped dict part and confirm each returns HTTP 400 before media fetch/download/worker dispatch.
5. Run a known-good Kimi K2.6/Kimi K2.5 video upload/chat smoke separately to verify video behavior remains intact.

## 2026-05-05 runtime smoke continuation

Follow-up after the clean live integration commit:

- Full MiMo Pro artifacts were found locally at `/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf/XiaomiMiMo--MiMo-V2.5-Pro` (`du -sh`: `962G`).
- A no-download/offline exo API smoke was attempted with `EXO_MODELS_READ_ONLY_DIRS=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf`, `EXO_OFFLINE=true`, no TurboQuant settings, and an isolated smoke home/port.
- First smoke attempt failed before API readiness because `EXO_DASHBOARD_DIR=dashboard/build` resolves relative to `$HOME`; it expected `/Users/studio2/dashboard/build`.
- Second smoke attempt used `EXO_DASHBOARD_DIR=exo/dashboard/build` and reached the API. It confirmed the Pro model card was available and generated placement previews, but with only one Studio2 node visible in this isolated smoke:
  - node RAM available: about `527,649,112,064` bytes
  - Pro storage size before correction: `1,250,000,000,000` bytes
  - placement previews: Pipeline/Ring, Pipeline/Jaccl, Tensor/Ring, Tensor/Jaccl all returned `No cycles found with sufficient memory`
- No model weights were loaded, no inference was run, no download was started, and no live service was left running.

Runtime integration fixes added after the smoke evidence:

- Register MLX-LM model type alias `mimo_v2 -> mimo_v2_flash` in `src/exo/worker/engines/mlx/utils_mlx.py`, because the local MiMo Pro `config.json` uses `model_type: mimo_v2` while the pinned MLX-LM runtime module is `mlx_lm.models.mimo_v2_flash`.
- Correct the MiMo Pro model-card storage estimate from `1,250,000,000,000` bytes to the actual local safetensors index total `1,033,369,538,304` bytes. This still cannot fit on one 512 GiB Studio node, but should permit placement only when enough cluster nodes are visible.
- Add a no-weight-load runtime regression test at `src/exo/worker/engines/mlx/tests/test_mimo_mlx_runtime.py` covering the alias and local config compatibility with the pinned MLX-LM MiMo V2 implementation.

Verification command:

```bash
EXO_DASHBOARD_DIR=exo/dashboard/build \
EXO_MODELS_READ_ONLY_DIRS=/Volumes/GLM5-NVMe/exo/mimo-v25-pro/hf \
EXO_OFFLINE=true \
uv run pytest \
  src/exo/worker/engines/mlx/tests/test_mimo_mlx_runtime.py \
  src/exo/shared/tests/test_mimo_v25_pro_model_card.py \
  src/exo/api/tests/test_mimo_v25_pro_chat_adapter.py \
  src/exo/api/tests/test_video_uploads.py \
  -q
```

Result:

```text
29 passed in 2.35s
```

Ruff and basedpyright were also clean for the new runtime/test files.

Current online status: MiMo Pro is still not online. The next safe runtime step is to start the actual multi-node exo cluster, confirm at least enough nodes are visible to cover ~1.033 TB model storage/RAM budget, then create a Pipeline or Tensor instance from `/instance/previews` and run one tiny text-only `/v1/chat/completions` request. Do not use TurboQuant for this path.
