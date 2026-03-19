from __future__ import annotations

from typing import Any

import numpy as np


NEG_INF = -1e15


def _clean_token(token: str) -> str:
    return token.replace("|", " ").strip()


def tokenize_alignment_text(processor: Any, text: str) -> list[int]:
    encoded = processor.tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def decode_greedy_segments(
    token_ids: np.ndarray,
    frame_step_seconds: float,
    tokenizer: Any,
    blank_id: int,
) -> list[dict[str, float | str | int]]:
    segments: list[dict[str, float | str | int]] = []
    current_token_id: int | None = None
    start_frame = 0

    def emit_segment(token_id: int | None, start: int, end: int) -> None:
        if token_id is None or token_id == blank_id or end <= start:
            return
        label = _clean_token(tokenizer.convert_ids_to_tokens(int(token_id)))
        if not label:
            return
        segments.append(
            {
                "label": label,
                "token_id": int(token_id),
                "start_frame": start,
                "end_frame": end,
                "start_sec": round(start * frame_step_seconds, 6),
                "end_sec": round(end * frame_step_seconds, 6),
                "duration_sec": round((end - start) * frame_step_seconds, 6),
            }
        )

    for frame_index, token_id in enumerate(token_ids.tolist()):
        if current_token_id is None:
            current_token_id = int(token_id)
            start_frame = frame_index
            continue
        if int(token_id) != current_token_id:
            emit_segment(current_token_id, start_frame, frame_index)
            current_token_id = int(token_id)
            start_frame = frame_index

    emit_segment(current_token_id, start_frame, len(token_ids))
    return segments


def ctc_viterbi_align(
    log_probs: np.ndarray,
    target_ids: list[int],
    blank_id: int,
    frame_step_seconds: float,
    tokenizer: Any,
) -> dict[str, Any]:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must be shaped [time, vocab]")
    if not target_ids:
        raise ValueError("target_ids must not be empty for forced alignment")

    extended = [blank_id]
    for token_id in target_ids:
        extended.extend([token_id, blank_id])
    time_steps = log_probs.shape[0]
    states = len(extended)

    dp = np.full((time_steps, states), NEG_INF, dtype=np.float64)
    backpointers = np.full((time_steps, states), -1, dtype=np.int32)

    dp[0, 0] = log_probs[0, blank_id]
    if states > 1:
        dp[0, 1] = log_probs[0, extended[1]]

    for time_index in range(1, time_steps):
        for state_index in range(states):
            candidates = [(dp[time_index - 1, state_index], state_index)]
            if state_index - 1 >= 0:
                candidates.append((dp[time_index - 1, state_index - 1], state_index - 1))
            if (
                state_index - 2 >= 0
                and extended[state_index] != blank_id
                and extended[state_index] != extended[state_index - 2]
            ):
                candidates.append((dp[time_index - 1, state_index - 2], state_index - 2))

            best_score, best_prev_state = max(candidates, key=lambda item: item[0])
            dp[time_index, state_index] = best_score + log_probs[time_index, extended[state_index]]
            backpointers[time_index, state_index] = best_prev_state

    final_candidates = [(dp[-1, states - 1], states - 1)]
    if states > 1:
        final_candidates.append((dp[-1, states - 2], states - 2))
    best_path_score, best_final_state = max(final_candidates, key=lambda item: item[0])

    state_path = np.empty(time_steps, dtype=np.int32)
    state_path[-1] = best_final_state
    for time_index in range(time_steps - 1, 0, -1):
        state_path[time_index - 1] = backpointers[time_index, state_path[time_index]]

    segments: list[dict[str, float | str | int]] = []
    start_frame = 0
    current_state = int(state_path[0])
    for time_index in range(1, time_steps + 1):
        boundary = time_index == time_steps or int(state_path[time_index]) != current_state
        if not boundary:
            continue
        token_id = int(extended[current_state])
        if token_id != blank_id:
            label = _clean_token(tokenizer.convert_ids_to_tokens(token_id))
            if label:
                segments.append(
                    {
                        "label": label,
                        "token_id": token_id,
                        "state_index": current_state,
                        "start_frame": start_frame,
                        "end_frame": time_index,
                        "start_sec": round(start_frame * frame_step_seconds, 6),
                        "end_sec": round(time_index * frame_step_seconds, 6),
                        "duration_sec": round((time_index - start_frame) * frame_step_seconds, 6),
                    }
                )
        if time_index < time_steps:
            current_state = int(state_path[time_index])
            start_frame = time_index

    return {
        "best_path_score": float(best_path_score),
        "state_path": state_path.tolist(),
        "segments": segments,
    }
