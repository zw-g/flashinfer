import json
from pathlib import Path

import pytest
import torch

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper


@pytest.mark.cpu_only
@pytest.mark.parametrize("definition_source", ["generated", "checked_in"])
@pytest.mark.parametrize("stage", ["decode", "prefill"])
@pytest.mark.parametrize(
    "kv_indptr,kv_last_page_len,qo_indptr",
    [
        ([0, 1], [1], [0, 1]),
        ([0, 1], [16], [0, 1]),
        ([0, 2], [1], [0, 1]),
        ([0, 2, 3], [3, 7], [0, 2, 4]),
    ],
    ids=["one_token", "full_page", "two_pages", "ragged_batch"],
)
def test_gqa_paged_reference_honors_kv_last_page_len(
    definition_source, stage, kv_indptr, kv_last_page_len, qo_indptr
):
    if stage == "decode":
        qo_indptr = list(range(len(kv_last_page_len) + 1))  # one query per request
    q = torch.zeros(qo_indptr[-1], 32, 128, dtype=torch.bfloat16)
    k_cache = torch.zeros(kv_indptr[-1], 16, 8, 128, dtype=torch.bfloat16)
    v_cache = torch.full_like(k_cache, 16)
    v_cache[:, 0] = 0
    if definition_source == "generated":
        wrapper = (
            BatchDecodeWithPagedKVCacheWrapper
            if stage == "decode"
            else BatchPrefillWithPagedKVCacheWrapper
        )
        definition = wrapper.run.fi_trace(q=q, paged_kv_cache=(k_cache, v_cache))
    else:
        trace_dir = Path(__file__).resolve().parents[2] / "trace/fi_trace_out"
        definition = json.loads(
            (trace_dir / f"gqa_paged_{stage}_h32_kv8_d128_ps16.json").read_text()
        )
    namespace = {}
    exec(definition["reference"], namespace)
    qo = [torch.tensor(qo_indptr, dtype=torch.int32)] if stage == "prefill" else []
    kv = [
        torch.tensor(x, dtype=torch.int32)
        for x in (kv_indptr, range(kv_indptr[-1]), kv_last_page_len)
    ]
    output, lse = namespace[f"_gqa_paged_{stage}_reference"](
        q, k_cache, v_cache, *qo, *kv, 1.0
    )

    # Q and K are zero, so every visible token gets the same weight, and V is 0
    # in the first slot of each page and 16 in the others. Under the causal mask
    # query i of q_len sees the first n = valid - q_len + 1 + i valid tokens of
    # its request: the output is 16 * (n - ceil(n / 16)) / n and the LSE log2(n).
    n = []
    for b, last_page_len in enumerate(kv_last_page_len):
        valid = (kv_indptr[b + 1] - kv_indptr[b] - 1) * 16 + last_page_len
        q_len = qo_indptr[b + 1] - qo_indptr[b]
        n += range(valid - q_len + 1, valid + 1)
    n = torch.tensor(n, dtype=torch.float64)
    expected_output = (16 * (n - (n / 16).ceil()) / n).to(torch.bfloat16)
    torch.testing.assert_close(
        output, expected_output[:, None, None].expand_as(output), rtol=0, atol=0
    )
    torch.testing.assert_close(lse, n.log2().float()[:, None].expand_as(lse))
