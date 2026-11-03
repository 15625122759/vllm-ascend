# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm_ascend.worker.v2.sample import apply_top_k_top_p as topk_op


def _logits() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])


def test_uses_triton_op_when_enabled():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(
            topk_op,
            "get_ascend_config",
            return_value=SimpleNamespace(
                enable_triton_apply_topk_topp=True,
                triton_topk_topp_disable_gate=True,
                triton_topk_topp_disable_no_sort=False,
            ),
        ),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(topk_op.logger, "info_once"),
    ):
        out = topk_op.apply_top_k_top_p_npu(logits, k, p)

    triton_op.assert_called_once_with(logits, k, p, top_p_mode="full_vocab", k_max=None, no_sort=True)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_falls_back_to_pytorch_when_disabled():
    logits = _logits()
    k = torch.tensor([1, 2])
    expected = logits.clone()

    with (
        patch.object(
            topk_op,
            "get_ascend_config",
            return_value=SimpleNamespace(
                enable_triton_apply_topk_topp=False,
                triton_topk_topp_disable_gate=False,
                triton_topk_topp_disable_no_sort=False,
            ),
        ),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p_npu(logits, k, None)

    pytorch_op.assert_called_once_with(logits, k, None)
    triton_op.assert_not_called()
    assert out is expected


def test_falls_back_to_pytorch_without_triton():
    logits = _logits()
    expected = logits.clone()

    with (
        patch.object(
            topk_op,
            "get_ascend_config",
            return_value=SimpleNamespace(
                enable_triton_apply_topk_topp=True,
                triton_topk_topp_disable_gate=True,
                triton_topk_topp_disable_no_sort=False,
            ),
        ),
        patch.object(topk_op, "HAS_TRITON", False),
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch", return_value=expected) as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p_npu(logits, None, torch.tensor([0.9, 0.9]))

    pytorch_op.assert_called_once()
    triton_op.assert_not_called()
    assert out is expected


@pytest.mark.skipif(
    "has_topp_only_rows" not in inspect.signature(topk_op.apply_top_k_top_p_npu).parameters,
    reason="has_topp_only_rows exists only on the vLLM main lane",
)
def test_accepts_has_topp_only_rows_on_main_lane():
    logits = _logits()
    k = torch.tensor([1, 2])
    p = torch.tensor([0.9, 0.9])
    triton_out = torch.full_like(logits, float("-inf"))

    with (
        patch.object(
            topk_op,
            "get_ascend_config",
            return_value=SimpleNamespace(
                enable_triton_apply_topk_topp=True,
                triton_topk_topp_disable_gate=True,
                triton_topk_topp_disable_no_sort=False,
            ),
        ),
        patch.object(topk_op, "HAS_TRITON", True),
        patch.object(topk_op, "is_triton_apply_topk_topp_supported", return_value=True),
        patch.object(topk_op, "triton_apply_top_k_top_p", return_value=triton_out) as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
        patch.object(topk_op.logger, "info_once"),
    ):
        out = topk_op.apply_top_k_top_p_npu(logits, k, p, has_topp_only_rows=False)

    # The hint is accepted (newer vLLM main passes it) but not forwarded:
    # the Triton op handles per-row k/p without a split pipeline.
    triton_op.assert_called_once_with(logits, k, p, top_p_mode="full_vocab", k_max=None, no_sort=True)
    pytorch_op.assert_not_called()
    assert torch.equal(out, triton_out)


def test_returns_logits_untouched_without_k_and_p():
    logits = _logits()

    with (
        patch.object(topk_op, "get_ascend_config") as config,
        patch.object(topk_op, "triton_apply_top_k_top_p") as triton_op,
        patch.object(topk_op, "_apply_top_k_top_p_pytorch") as pytorch_op,
    ):
        out = topk_op.apply_top_k_top_p_npu(logits, None, None)

    config.assert_not_called()
    triton_op.assert_not_called()
    pytorch_op.assert_not_called()
    assert out is logits


def test_get_top_k_top_p_with_hint_publishes_k_max():
    states = SimpleNamespace(
        vocab_size=8,
        top_k=SimpleNamespace(np=np.array([2, 4]), gpu=torch.tensor([2, 4])),
        top_p=SimpleNamespace(np=np.array([0.9, 0.9]), gpu=torch.tensor([0.9, 0.9])),
    )
    expanded_idx_mapping = torch.tensor([0, 1])
    idx_mapping_np = np.array([0, 1])
    params = inspect.signature(topk_op.get_top_k_top_p_with_hint).parameters

    with patch.object(topk_op.logger, "info_once"):
        if "vocab_size" in params:
            out = topk_op.get_top_k_top_p_with_hint(states, expanded_idx_mapping, idx_mapping_np, states.vocab_size)
            assert len(out) == 3
            assert out[2] is False
        else:
            out = topk_op.get_top_k_top_p_with_hint(states, expanded_idx_mapping, idx_mapping_np)
            assert len(out) == 2

    assert topk_op._take_k_max_hint() == 4
