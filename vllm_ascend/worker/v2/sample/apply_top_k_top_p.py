import numpy as np
import torch
from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.triton.sample.apply_top_k_top_p import (
    apply_top_k_top_p as triton_apply_top_k_top_p,
)
from vllm_ascend.ops.triton.sample.apply_top_k_top_p import (
    is_triton_apply_topk_topp_supported,
    is_triton_topk_topp_profitable,
)
from vllm_ascend.utils import vllm_version_is

# V2 keeps top_k in a UVA-backed tensor whose CPU mirror (``top_k.np``) is
# readable without a device sync. The patched SamplingStates.get_top_k_top_p
# publishes max(k) here; apply_top_k_top_p_npu consumes it once, so callers
# that skip the getter fall back to the device-sync path.
_K_MAX_HINT: int | None = None


def _set_k_max_hint(k_max: int | None) -> None:
    global _K_MAX_HINT
    _K_MAX_HINT = k_max


def _take_k_max_hint() -> int | None:
    global _K_MAX_HINT
    hint = _K_MAX_HINT
    _K_MAX_HINT = None
    return hint


def _read_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np, vocab_size):
    """Read per-request k/p from the CPU mirror and publish max(k)."""
    k_np = self.top_k.np[idx_mapping_np]
    p_np = self.top_p.np[idx_mapping_np]
    do_top_k = bool(np.any(k_np != self.vocab_size))
    do_top_p = bool(np.any(p_np != 1.0))
    top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
    top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
    _set_k_max_hint(int(k_np.max()) if do_top_k else None)
    logger.info_once(
        "[sample/v2] k_max read from the UVA CPU mirror (no host sync): max_k=%d.",
        int(k_np.max()) if do_top_k else self.vocab_size,
    )
    has_topp_only_rows = bool(do_top_p and np.any((k_np >= vocab_size) & (p_np < 1.0)))
    return top_k, top_p, has_topp_only_rows


# vLLM main added `vocab_size` and the `has_topp_only_rows` result; the
# 0.32.0 release still uses the two-argument getter.
if vllm_version_is("0.32.0"):

    def get_top_k_top_p_with_hint(self, expanded_idx_mapping, idx_mapping_np):
        top_k, top_p, _ = _read_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np, self.vocab_size)
        return top_k, top_p

else:

    def get_top_k_top_p_with_hint(  # type: ignore[misc]
        self, expanded_idx_mapping, idx_mapping_np, vocab_size
    ):
        return _read_top_k_top_p(self, expanded_idx_mapping, idx_mapping_np, vocab_size)


# Canonical sort-based top-k/top-p masking implementation.
# It lives under the MRv2 worker because vllm_ascend.sample is MRv1-only and is
# scheduled for deletion: MRv1 imports it from here, so the dependency only ever
# points from the dying code to the surviving code.
def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,  # [B, V_local]
    k: torch.Tensor,  # [B] or None
    p: torch.Tensor,  # [B] or None
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _log_pytorch_fallback(logits: torch.Tensor) -> None:
    """Log the exercised V2 path explicitly (needed for A/B traces)."""
    logger.info_once(
        "[sample/v2] Using the PyTorch apply_top_k_top_p fallback "
        "(stock sort-based path, enable_triton_apply_topk_topp=%s): logits shape %s.",
        get_ascend_config().enable_triton_apply_topk_topp,
        tuple(logits.shape),
    )


def _apply_top_k_top_p_npu_impl(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    # Consume the hint emitted by the patched SamplingStates getter before
    # any early return, so a stale hint can never leak into a later call.
    k_max = _take_k_max_hint()
    if k is None and p is None:
        return logits
    cfg = get_ascend_config()
    if cfg.enable_triton_apply_topk_topp and HAS_TRITON:
        if not is_triton_apply_topk_topp_supported():
            logger.warning_once(
                "[sample/v2] Triton apply_top_k_top_p is not supported on this device; "
                "falling back to the PyTorch path.",
            )
        else:
            B, V = logits.shape
            if not is_triton_topk_topp_profitable(B, V, disable_gate=cfg.triton_topk_topp_disable_gate):
                logger.warning_once(
                    "[sample/v2] Triton apply_top_k_top_p skipped (B=%d V=%d below the "
                    "measured performance crossover); using the PyTorch path.",
                    B,
                    V,
                )
            else:
                logger.info_once(
                    "[sample/v2] Using Triton apply_top_k_top_p op (full_vocab semantics) "
                    "for top-k/top-p masking: logits shape %s.",
                    tuple(logits.shape),
                )
                return triton_apply_top_k_top_p(
                    logits,
                    k,
                    p,
                    top_p_mode="full_vocab",
                    k_max=k_max,
                    no_sort=not cfg.triton_topk_topp_disable_no_sort,
                )
    _log_pytorch_fallback(logits)
    return _apply_top_k_top_p_pytorch(logits, k, p)


# NOTE: During the warmup stage, if both k and p are None, the NPU op is skipped.
# As a result, its workspace is not reserved during warmup, which may lead to an OOM
# when the op is executed later.
# TODO: Fix the workspace reservation issue in the future.
if vllm_version_is("0.32.0"):

    def apply_top_k_top_p_npu(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
        return _apply_top_k_top_p_npu_impl(logits, k, p)

else:
    # vLLM main added `has_topp_only_rows` to select the split top-p Triton
    # pipeline. The Triton op and the PyTorch reference both handle per-row
    # k/p (including top-p-only rows), so the value is accepted but unused.
    def apply_top_k_top_p_npu(  # type: ignore[misc]
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        has_topp_only_rows: bool = True,
    ) -> torch.Tensor:
        return _apply_top_k_top_p_npu_impl(logits, k, p)
