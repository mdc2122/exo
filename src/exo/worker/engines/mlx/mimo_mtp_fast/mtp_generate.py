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
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.sample_utils import (
    apply_min_p as _apply_min_p,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.sample_utils import (
    apply_top_k as _apply_top_k,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.sample_utils import (
    apply_top_p as _apply_top_p,  # pyright: ignore[reportUnknownVariableType]
)
from mlx_lm.sample_utils import (
    make_logits_processors,
    make_sampler,
)
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
from exo.worker.engines.mlx.cache import (
    encode_prompt,
    has_non_kv_caches,
    make_kv_cache,
    trim_cache,
)
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


def _sampling_probabilities(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
) -> mx.array:
    """Return the normalized sampling distribution for logits."""
    if temperature == 0.0:
        # Greedy distribution is a one-hot at the argmax. Top-p/min-p/top-k
        # filters can never remove the argmax token, so skip them — they cost
        # full-vocabulary sorts on the hot path.
        greedy_token = mx.argmax(logits, axis=-1, keepdims=True)
        return mx.where(
            mx.arange(logits.shape[-1])[None, :] == greedy_token,
            mx.ones_like(logits),
            mx.zeros_like(logits),
        )

    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    if top_p > 0.0 and top_p < 1.0:
        logprobs = cast(mx.array, _apply_top_p(logprobs, top_p))
    if min_p != 0.0:
        logprobs = cast(mx.array, _apply_min_p(logprobs, min_p))
    if top_k > 0:
        logprobs = cast(mx.array, _apply_top_k(logprobs, top_k))
    return mx.softmax(logprobs / temperature, axis=-1)


def _token_probability(probabilities: mx.array, token_id: int | mx.array) -> mx.array:
    if isinstance(token_id, mx.array):
        token_index = token_id.reshape(1, 1).astype(mx.int32)
    else:
        token_index = mx.array([[token_id]], dtype=mx.int32)
    return mx.take_along_axis(probabilities, token_index, axis=-1)[0, 0]


def _acceptance_probability_from_probabilities(
    *,
    target_probabilities: mx.array,
    draft_probabilities: mx.array,
    draft_token: int | mx.array,
) -> mx.array:
    """Compute min(1, p_target(draft_token) / q_draft(draft_token))."""
    p_target = _token_probability(target_probabilities, draft_token)
    q_draft = _token_probability(draft_probabilities, draft_token)
    return mx.minimum(mx.array(1.0), p_target / mx.maximum(q_draft, mx.array(1e-20)))


def _residual_correction_from_probabilities(
    *,
    target_probabilities: mx.array,
    draft_probabilities: mx.array,
) -> mx.array:
    """Sample from normalized positive residual max(p_target - q_draft, 0).

    Returns a lazy scalar token array so the caller can fold materialization
    into a single mx.eval alongside the acceptance test.
    """
    residual = mx.maximum(target_probabilities - draft_probabilities, mx.zeros_like(target_probabilities))
    residual_total = mx.sum(residual, axis=-1, keepdims=True)
    corrected_probabilities = mx.where(
        residual_total > 0.0,
        residual / residual_total,
        target_probabilities,
    )
    corrected_logits = mx.where(
        corrected_probabilities > 0.0,
        mx.log(corrected_probabilities),
        mx.array(-float("inf"), dtype=corrected_probabilities.dtype),
    )
    return mx.random.categorical(corrected_logits)


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
    requested_mtp_depth: int | None,
    mtp_sidecar_status: str | None,
    fallback_count: int,
    attempted_depth_counts: dict[str, int],
    accepted_depth_counts: dict[str, int],
    timing_accumulators: dict[str, float] | None = None,
) -> GenerationStats:
    elapsed = time.perf_counter() - generation_start_time
    generation_tps = generated_tokens / elapsed if elapsed > 0 else 0.0
    attempted_draft_tokens = sum(
        count for depth, count in attempted_depth_counts.items() if depth != "0"
    )
    accepted_draft_tokens = sum(
        count for depth, count in accepted_depth_counts.items() if depth != "0"
    )
    acceptance_rate = (
        accepted_draft_tokens / attempted_draft_tokens
        if attempted_draft_tokens > 0
        else 0.0
    )
    return GenerationStats(
        prompt_tps=float(prefill_tps),
        generation_tps=float(generation_tps),
        prompt_tokens=0,
        generation_tokens=generated_tokens,
        peak_memory_usage=Memory.from_gb(0.0),
        accepted_execution_path=accepted_execution_path,
        mtp_enabled=mtp_enabled,
        requested_mtp_depth=requested_mtp_depth,
        mtp_depth=mtp_depth,
        mtp_sidecar_status=mtp_sidecar_status,
        attempted_depth_counts=attempted_depth_counts,
        accepted_depth_counts=accepted_depth_counts,
        acceptance_rate=acceptance_rate,
        fallback_count=fallback_count,
        timing_breakdown_seconds={
            "generation": float(elapsed),
            **(timing_accumulators or {}),
        },
    )


def _encode_text(text: str, tokenizer: TokenizerWrapper) -> list[int]:
    """Encode text to token IDs for logits processor history."""
    if not text:
        return []
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return []


def _decode_tokens(token_ids: list[int], tokenizer: TokenizerWrapper) -> str:
    """Decode token IDs while preserving a strict string type for pyright."""
    return tokenizer.decode(token_ids)


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
    mtp_depth: int = min(decision.mtp_depth, 1)  # Cap at 1 for MiMo

    all_prompt_tokens = encode_prompt(tokenizer, prompt)
    target_cache = make_kv_cache(model=model)

    # MiMo's sliding-window layers get RotatingKVCache from make_cache. The
    # window is enforced by the attention mask (create_attention_mask receives
    # window_size explicitly, and emits a windowed array mask even for N=1);
    # the ring buffer only bounds memory. Substituting a plain KVCache keeps
    # attention bit-identical while making every entry single-token trimmable,
    # which the batched verify's rejection rewind requires. RotatingKVCache
    # cannot rewind: a 2-token update takes the concat path and a subsequent
    # trim corrupts _temporal_order's ring bookkeeping. Memory cost is the
    # full-length K/V for window layers — negligible at MTP context lengths.
    target_cache = [
        KVCache() if isinstance(entry, RotatingKVCache) else entry
        for entry in target_cache
    ]

    # Batched verify commits [primary, draft] in one target forward and rewinds
    # the cache by one token on rejection. Rewind via trim is only sound for
    # plain KV caches; SSM layers cannot trim a single token, so fall back to
    # autoregressive decoding (drafting disabled) for such models.
    can_batch_verify = not has_non_kv_caches(target_cache)
    if not can_batch_verify:
        logger.warning(
            "mlx_generate_mtp: cache has non-KV layers; MTP drafting disabled, "
            "running autoregressive fallback"
        )

    temperature = task.temperature if task.temperature is not None else 0.7
    top_p = task.top_p if task.top_p is not None else 1.0
    min_p = task.min_p if task.min_p is not None else 0.05
    top_k = task.top_k if task.top_k is not None else 0
    sampler: Callable[[mx.array], mx.array] = make_sampler(temp=temperature, top_p=top_p, min_p=min_p, top_k=top_k)

    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = make_logits_processors(
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
    current_hidden: mx.array = hidden_state[:, -1:, :]
    current_logits: mx.array = logits[:, -1, :]

    # --- MTP DECODE LOOP ---
    generated_tokens = 0
    accumulated_text: str = ""
    max_stop_len = max((len(s) for s in stop_sequences), default=0)
    pending_primary: int | None = None
    attempted_depth_counts: dict[str, int] = {}
    accepted_depth_counts: dict[str, int] = {}
    accepted_drafts = 0
    rejected_drafts = 0
    fallback_count = 0
    timing_accumulators: dict[str, float] = {
        "draft_seconds": 0.0,
        "verify_build_seconds": 0.0,
        "verify_eval_seconds": 0.0,
    }
    generation_start_time = time.perf_counter()

    while generated_tokens < max_tokens:
        # --- SAMPLE_PRIMARY ---
        if pending_primary is not None:
            primary_token = pending_primary
            pending_primary = None
        else:
            processed_logits = current_logits
            for processor in logits_processors:  # pyright: ignore[reportUnknownVariableType]
                history_tokens = mx.array(list(all_prompt_tokens) + _encode_text(accumulated_text, tokenizer))
                processed_logits = processor(history_tokens, processed_logits)  # pyright: ignore[reportUnknownVariableType]
            primary_token = int(sampler(processed_logits).item())

        text = _decode_tokens([primary_token], tokenizer)
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
                requested_mtp_depth=decision.requested_depth,
                mtp_sidecar_status=decision.sidecar_status,
                fallback_count=fallback_count,
                attempted_depth_counts=attempted_depth_counts, accepted_depth_counts=accepted_depth_counts,
                timing_accumulators=timing_accumulators,
            )

        if on_generation_token is not None:
            on_generation_token()

        yield GenerationResponse(text=text, token=primary_token, logprob=None, top_logprobs=None, finish_reason=finish_reason, stats=stats, usage=usage)

        if is_done:
            break

        # --- DRAFT_PROPOSE ---
        mtp_cache = mtp_stack.make_cache()
        draft_token: int | None = None
        draft_token_array: mx.array | None = None

        draft_logits_for_acceptance: mx.array | None = None
        draft_section_start = time.perf_counter()
        if mtp_depth >= 1 and can_batch_verify:
            try:
                primary_batch = _token_batch(primary_token)
                processed_draft_logits: list[mx.array] = []

                def draft_sampler(
                    draft_logits: mx.array,
                    *,
                    accumulated_text_snapshot: str = accumulated_text,
                    processed_draft_logits_sink: list[mx.array] = processed_draft_logits,
                ) -> mx.array:
                    processed_logits = draft_logits
                    for processor in logits_processors:
                        draft_history = mx.array(list(all_prompt_tokens) + _encode_text(accumulated_text_snapshot, tokenizer))
                        processed_logits = processor(draft_history, processed_logits)
                    processed_draft_logits_sink.append(processed_logits)
                    return sampler(processed_logits)

                draft_tokens, raw_draft_logits = mtp_stack.propose(
                    previous_hidden_state=current_hidden,
                    latest_token_ids=primary_batch,
                    max_draft_tokens=mtp_depth,  # pyright: ignore[reportUnknownArgumentType]
                    sampler=draft_sampler,
                    cache=mtp_cache,
                )
                # Keep the draft token lazy: it feeds the verify batch as an
                # array, so the draft and verify graphs evaluate together in
                # the single per-iteration mx.eval. Only the (uncommon)
                # logits-processor path needs the token id on the CPU early,
                # because processor history must include the draft text.
                draft_token_array = draft_tokens[:, :1]
                if logits_processors:
                    mx.eval(draft_tokens)
                draft_token = int(draft_tokens[0, 0].item()) if logits_processors else None
                draft_logits_for_acceptance = (
                    processed_draft_logits[0]
                    if processed_draft_logits
                    else raw_draft_logits[0]
                )
            except Exception as exc:
                logger.warning("MTP draft proposal failed, AR fallback: {}", exc)
                draft_token_array = None
        timing_accumulators["draft_seconds"] += time.perf_counter() - draft_section_start

        if draft_token_array is None or draft_logits_for_acceptance is None:
            # AR fallback: forward primary through target model
            fallback_count += 1
            primary_batch = _token_batch(primary_token)
            ar_hidden, ar_logits = _forward_hidden_and_logits(
                model, primary_batch, target_cache
            )
            mx.eval(ar_hidden, ar_logits)
            current_logits = ar_logits[:, -1, :]
            current_hidden = ar_hidden[:, -1:, :]
            continue

        # --- BATCHED VERIFY_FORWARD ---
        # One target forward commits BOTH primary and draft to the KV cache.
        # Position 0's logits verify the draft; position 1's logits supply the
        # bonus distribution on acceptance. On rejection the draft is trimmed
        # back out of the cache. This is what makes speculation pay: 1 + accept
        # tokens per target forward instead of one forward per token.
        verify_build_start = time.perf_counter()
        verify_batch = mx.concatenate(
            [_token_batch(primary_token), draft_token_array.astype(mx.int32)], axis=1
        )
        verify_hidden, verify_logits = _forward_hidden_and_logits(
            model, verify_batch, target_cache
        )
        target_logits_for_draft = verify_logits[:, 0, :]
        bonus_logits = verify_logits[:, 1, :]
        processed_target_logits_for_draft: mx.array = target_logits_for_draft
        for processor in logits_processors:
            draft_history = mx.array(list(all_prompt_tokens) + _encode_text(accumulated_text, tokenizer))
            processed_target_logits_for_draft = processor(draft_history, processed_target_logits_for_draft)

        # --- COMPUTE_ACCEPTANCE (p/q speculative sampling) ---
        # The draft, verify, and both branch outcomes (bonus token on accept,
        # residual correction on reject) form one lazy graph materialized by a
        # single mx.eval — one GPU sync per iteration.
        target_probabilities = _sampling_probabilities(
            processed_target_logits_for_draft,
            temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        draft_probabilities = _sampling_probabilities(
            draft_logits_for_acceptance,
            temperature=temperature, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        acceptance_probability = _acceptance_probability_from_probabilities(
            target_probabilities=target_probabilities,
            draft_probabilities=draft_probabilities,
            draft_token=draft_token if draft_token is not None else draft_token_array,
        )
        acceptance_draw = mx.random.uniform(shape=())

        processed_bonus: mx.array = bonus_logits
        for processor in logits_processors:
            # draft_token is materialized on this path (see DRAFT_PROPOSE)
            assert draft_token is not None
            bonus_history = mx.array(
                list(all_prompt_tokens)
                + _encode_text(
                    accumulated_text + _decode_tokens([draft_token], tokenizer),
                    tokenizer,
                )
            )
            processed_bonus = processor(bonus_history, processed_bonus)
        bonus_token_array = sampler(processed_bonus)

        correction_token_array = _residual_correction_from_probabilities(
            target_probabilities=target_probabilities,
            draft_probabilities=draft_probabilities,
        )

        verify_eval_start = time.perf_counter()
        timing_accumulators["verify_build_seconds"] += verify_eval_start - verify_build_start
        mx.eval(
            draft_token_array,
            acceptance_probability,
            acceptance_draw,
            bonus_token_array,
            correction_token_array,
            verify_hidden,
            processed_target_logits_for_draft,
        )
        timing_accumulators["verify_eval_seconds"] += time.perf_counter() - verify_eval_start
        if draft_token is None:
            draft_token = int(draft_token_array[0, 0].item())
        draft_text = _decode_tokens([draft_token], tokenizer)
        accepted = float(acceptance_draw.item()) <= float(acceptance_probability.item())

        if accepted:
            # --- ACCEPT_BRANCH ---
            accepted_drafts += 1
            _increment_count(attempted_depth_counts, 1)
            _increment_count(accepted_depth_counts, 1)

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
                    requested_mtp_depth=decision.requested_depth,
                    mtp_sidecar_status=decision.sidecar_status,
                    fallback_count=fallback_count,
                    attempted_depth_counts=attempted_depth_counts, accepted_depth_counts=accepted_depth_counts,
                    timing_accumulators=timing_accumulators,
                )

            if on_generation_token is not None:
                on_generation_token()

            yield GenerationResponse(text=draft_text, token=draft_token, logprob=None, top_logprobs=None, finish_reason=draft_finish_reason, stats=draft_stats, usage=draft_usage)

            if is_draft_done:
                break

            # Draft is already committed to the target cache by the batched
            # verify forward; its hidden state and the pre-sampled bonus token
            # carry the loop forward with no extra target forward.
            current_logits = bonus_logits
            current_hidden = verify_hidden[:, 1:2, :]

            # --- BONUS (pre-sampled during the verify eval) ---
            if generated_tokens < max_tokens:
                pending_primary = int(bonus_token_array.item())
        else:
            # --- REJECT ---
            rejected_drafts += 1
            _increment_count(attempted_depth_counts, 1)
            # Rewind the speculatively committed draft token out of the cache.
            trim_cache(target_cache, 1)
            current_logits = processed_target_logits_for_draft
            current_hidden = verify_hidden[:, 0:1, :]
            pending_primary = int(correction_token_array.item())

        if max_stop_len > 0 and len(accumulated_text) > max_stop_len:
            accumulated_text = accumulated_text[-max_stop_len:]
