"""MiMo V2.5 Pro MTP speculative decoding generation function.

Implements the real MTP depth=1 speculative decoding loop for exo's
distributed MLX worker/generator path, following MTPLX generate_mtp1
(commit 0ad700c), adapted for exo's generation infrastructure.

Algorithm (depth=1, sequential verify strategy):
  1. PREFILL target model → populate KV cache, get initial logits + hidden.
  2. SAMPLE_PRIMARY from target logits.
  3. DRAFT_PROPOSE: MtpStack.propose with hidden state + primary token.
  4. VERIFY_FORWARD: forward primary through target to get verify logits.
  5. ACCEPT if draft_token == argmax(target_logits); else REJECT.
  6. On accept: emit draft, forward draft through target, BONUS_SAMPLE.
  7. On reject: use verify logits/hidden for next iteration.
  8. EOS/stop/max_tokens handled consistently with AR semantics.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Generator
from typing import cast

import mlx.core as mx
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.api.types import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import encode_prompt, make_kv_cache
from exo.worker.engines.mlx.generator.generate import (
    eos_ids_from_tokenizer,
    prefill,
)
from exo.worker.engines.mlx.mimo_mtp_fast.sidecar_module import (
    build_mimo_mtp_stack,
)
from exo.worker.engines.mlx.mimo_mtp_fast.worker_fastpath import (
    MimoMtpWorkerFastpathDecision,
)
from exo.worker.runner.bootstrap import logger


def _forward_hidden_and_logits(
    model: Model, token_ids: mx.array, cache: KVCacheType
) -> tuple[mx.array, mx.array]:
    """Commit token_ids to cache once and return both hidden states and logits.

    The top-level MLX language model call is usually equivalent to
    ``model.lm_head(model.model(token_ids, cache))``. Calling the top-level
    model and then calling ``model.model`` again with the same cache would append
    the same tokens twice. MTP verification uses this helper so the primary,
    accepted draft, rejected draft boundary, and following correction token each
    advance the target KV cache exactly once when they are actually committed.
    """
    inner = model.model  # type: ignore[reportAttributeAccessIssue]
    lm_head = model.lm_head  # type: ignore[reportAttributeAccessIssue]
    hidden = cast(mx.array, inner(token_ids, cache))
    logits = cast(mx.array, lm_head(hidden))
    return hidden, logits


def _token_batch(token_id: int) -> mx.array:
    """Create a (1,1) token array for single-token model forward."""
    return mx.array([[token_id]], dtype=mx.int32)


def _greedy_sampler(logits: mx.array) -> mx.array:
    """Greedy argmax sampler returning a single-column token ID array."""
    token_ids = mx.argmax(logits, axis=-1)
    if token_ids.ndim == 1:
        token_ids = token_ids[:, None]
    return token_ids.astype(mx.int32)


def _check_stop_sequence(accumulated_text: str, stop_sequences: list[str]) -> bool:
    """Check if any stop sequence appears in the accumulated text."""
    return any(seq in accumulated_text for seq in stop_sequences) if stop_sequences else False


def _increment_count(counts: dict[str, int], depth: int) -> None:
    """Increment a depth count in the telemetry dict (string keys)."""
    key = str(depth)
    counts[key] = counts.get(key, 0) + 1


def _build_usage(*, total_prompt_tokens: int, completion_tokens: int, prefix_hit_length: int) -> Usage:
    return Usage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=prefix_hit_length),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )


def _build_stats(
    *,
    prefill_tps: float,
    generation_start_time: float,
    generated_tokens: int,
    accepted_execution_path: str,
    mtp_enabled: bool,
    mtp_depth: int,
    attempted_depth_counts: dict[str, int],
    accepted_depth_counts: dict[str, int],
) -> GenerationStats:
    elapsed = time.perf_counter() - generation_start_time
    generation_tps = generated_tokens / elapsed if elapsed > 0 else 0.0
    return GenerationStats(
        prompt_tps=float(prefill_tps),
        generation_tps=float(generation_tps),
        prompt_tokens=0,
        generation_tokens=generated_tokens,
        peak_memory_usage=Memory.from_gb(0.0),
        accepted_execution_path=accepted_execution_path,
        mtp_enabled=mtp_enabled,
        mtp_depth=mtp_depth,
        attempted_depth_counts=attempted_depth_counts,
        accepted_depth_counts=accepted_depth_counts,
    )


def _encode_text(text: str, tokenizer: TokenizerWrapper) -> list[int]:
    """Encode text to token IDs for logits processor history."""
    if not text:
        return []
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return []


def mlx_generate_mtp(
    model: Model,
    tokenizer: TokenizerWrapper,
    task: TextGenerationTaskParams,
    prompt: str,
    decision: MimoMtpWorkerFastpathDecision,
    *,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
) -> Generator[GenerationResponse]:
    """Generate tokens using MiMo V2.5 Pro MTP speculative decoding.

    Only called when decision.should_use_mtp is True and all fastpath
    guards have passed. Uses sequential verify strategy for depth=1.
    """
    assert decision.should_use_mtp
    assert decision.sidecar is not None
    assert decision.mtp_depth is not None and decision.mtp_depth >= 1

    mtp_stack = build_mimo_mtp_stack(decision.sidecar, base_model=model)
    mtp_depth = min(decision.mtp_depth, 1)  # Cap at 1 for MiMo

    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    target_cache = make_kv_cache(model=model)

    temperature = task.temperature if task.temperature is not None else 0.7
    top_p = task.top_p if task.top_p is not None else 1.0
    min_p = task.min_p if task.min_p is not None else 0.05
    top_k = task.top_k if task.top_k is not None else 0
    sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

    logits_processors = make_logits_processors(
        repetition_penalty=task.repetition_penalty,
        repetition_context_size=task.repetition_context_size,
    )

    eos_ids = eos_ids_from_tokenizer(tokenizer)
    stop_sequences: list[str] = (
        ([task.stop] if isinstance(task.stop, str) else task.stop)
        if task.stop is not None else []
    )

    max_tokens = task.max_output_tokens or 100
    total_prompt_tokens = len(all_prompt_tokens)

    # --- PREFILL ---
    logger.info("mlx_generate_mtp entering prefill: prompt_tokens={}", len(all_prompt_tokens))
    prefill_tps, prefill_tokens, _ = prefill(
        model, tokenizer, sampler, all_prompt_tokens[:-1],
        target_cache, group=None,
        on_prefill_progress=on_prefill_progress,
        distributed_prompt_progress_callback=None,
    )
    logger.info("mlx_generate_mtp prefill finished: prefill_tokens={} prefill_tps={:.2f}", prefill_tokens, prefill_tps)

    # Forward last prompt tokens to get logits AND hidden state
    last_prompt_tokens = all_prompt_tokens[-2:]
    hidden_state, logits = _forward_hidden_and_logits(
        model, last_prompt_tokens[None], target_cache
    )
    mx.eval(hidden_state, logits)
    current_hidden = hidden_state[:, -1:, :]
    current_logits = logits[:, -1, :]

    # --- MTP DECODE LOOP ---
    generated_tokens = 0
    accumulated_text = ""
    max_stop_len = max((len(s) for s in stop_sequences), default=0)
    pending_primary: int | None = None
    attempted_depth_counts: dict[str, int] = {}
    accepted_depth_counts: dict[str, int] = {}
    accepted_drafts = 0
    rejected_drafts = 0
    generation_start_time = time.perf_counter()

    while generated_tokens < max_tokens:
        # --- SAMPLE_PRIMARY ---
        if pending_primary is not None:
            primary_token = pending_primary
            pending_primary = None
        else:
            processed_logits = current_logits
            for processor in logits_processors:
                history_tokens = mx.array(list(all_prompt_tokens) + _encode_text(accumulated_text, tokenizer))
                processed_logits = processor(history_tokens, processed_logits)
            primary_token = int(sampler(processed_logits).item())

        text = tokenizer.decode([primary_token])
        accumulated_text += text
        generated_tokens += 1
        _increment_count(attempted_depth_counts, 0)
        _increment_count(accepted_depth_counts, 0)

        # --- CHECK_BUDGET_OR_STOP ---
        is_eos = primary_token in eos_ids if eos_ids else False
        is_stop = _check_stop_sequence(accumulated_text, stop_sequences)
        is_done = is_eos or is_stop or generated_tokens >= max_tokens

        finish_reason: FinishReason | None = None
        if is_eos or is_stop:
            finish_reason = "stop"
        elif generated_tokens >= max_tokens:
            finish_reason = "length"

        usage = _build_usage(total_prompt_tokens=total_prompt_tokens, completion_tokens=generated_tokens, prefix_hit_length=0)
        stats: GenerationStats | None = None
        if is_done:
            stats = _build_stats(
                prefill_tps=prefill_tps, generation_start_time=generation_start_time,
                generated_tokens=generated_tokens, accepted_execution_path="mimo_mtp_fastpath",
                mtp_enabled=True, mtp_depth=mtp_depth,
                attempted_depth_counts=attempted_depth_counts, accepted_depth_counts=accepted_depth_counts,
            )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(text=text, token=primary_token, logprob=None, top_logprobs=None, finish_reason=finish_reason, stats=stats, usage=usage)

        if is_done:
            break

        # --- DRAFT_PROPOSE ---
        mtp_cache = mtp_stack.make_cache()
        draft_token: int | None = None

        if mtp_depth >= 1:
            try:
                primary_batch = _token_batch(primary_token)
                draft_logits, _ = mtp_stack.propose(
                    previous_hidden_state=current_hidden,
                    latest_token_ids=primary_batch,
                    max_draft_tokens=mtp_depth,
                    sampler=_greedy_sampler,
                    cache=mtp_cache,
                )
                mx.eval(draft_logits)
                draft_token = int(draft_logits[0, 0].item())
            except Exception as exc:
                logger.warning("MTP draft proposal failed, AR fallback: {}", exc)
                draft_token = None

        if draft_token is None:
            # AR fallback: forward primary through target model
            primary_batch = _token_batch(primary_token)
            ar_hidden, ar_logits = _forward_hidden_and_logits(
                model, primary_batch, target_cache
            )
            mx.eval(ar_hidden, ar_logits)
            current_logits = ar_logits[:, -1, :]
            current_hidden = ar_hidden[:, -1:, :]
            continue

        # --- VERIFY_FORWARD (sequential strategy) ---
        primary_batch = _token_batch(primary_token)
        verify_hidden, verify_logits = _forward_hidden_and_logits(
            model, primary_batch, target_cache
        )
        mx.eval(verify_logits, verify_hidden)
        target_logits_for_draft = verify_logits[:, -1, :]

        # --- COMPUTE_ACCEPTANCE (greedy) ---
        target_token = int(mx.argmax(target_logits_for_draft[0]).item())
        accepted = draft_token == target_token

        if accepted:
            # --- ACCEPT_BRANCH ---
            accepted_drafts += 1
            _increment_count(attempted_depth_counts, 1)
            _increment_count(accepted_depth_counts, 1)

            draft_text = tokenizer.decode([draft_token])
            accumulated_text += draft_text
            generated_tokens += 1

            is_draft_eos = draft_token in eos_ids if eos_ids else False
            is_draft_stop = _check_stop_sequence(accumulated_text, stop_sequences)
            is_draft_done = is_draft_eos or is_draft_stop or generated_tokens >= max_tokens

            draft_finish_reason: FinishReason | None = None
            if is_draft_eos or is_draft_stop:
                draft_finish_reason = "stop"
            elif generated_tokens >= max_tokens:
                draft_finish_reason = "length"

            draft_usage = _build_usage(total_prompt_tokens=total_prompt_tokens, completion_tokens=generated_tokens, prefix_hit_length=0)
            draft_stats: GenerationStats | None = None
            if is_draft_done:
                draft_stats = _build_stats(
                    prefill_tps=prefill_tps, generation_start_time=generation_start_time,
                    generated_tokens=generated_tokens, accepted_execution_path="mimo_mtp_fastpath",
                    mtp_enabled=True, mtp_depth=mtp_depth,
                    attempted_depth_counts=attempted_depth_counts, accepted_depth_counts=accepted_depth_counts,
                )

            if on_generation_token is not None:
                on_generation_token()

            yield GenerationResponse(text=draft_text, token=draft_token, logprob=None, top_logprobs=None, finish_reason=draft_finish_reason, stats=draft_stats, usage=draft_usage)

            if is_draft_done:
                break

            # Forward draft token through target to update KV cache
            draft_batch = _token_batch(draft_token)
            next_hidden, next_logits = _forward_hidden_and_logits(
                model, draft_batch, target_cache
            )
            mx.eval(next_logits, next_hidden)
            current_logits = next_logits[:, -1, :]
            current_hidden = next_hidden[:, -1:, :]

            # --- BONUS_SAMPLE ---
            if generated_tokens < max_tokens:
                processed_bonus = current_logits
                for processor in logits_processors:
                    bonus_history = mx.array(list(all_prompt_tokens) + _encode_text(accumulated_text, tokenizer))
                    processed_bonus = processor(bonus_history, processed_bonus)
                bonus_token = int(sampler(processed_bonus).item())
                pending_primary = bonus_token
        else:
            # --- REJECT ---
            rejected_drafts += 1
            _increment_count(attempted_depth_counts, 1)
            current_logits = target_logits_for_draft
            current_hidden = verify_hidden[:, -1:, :]

        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
