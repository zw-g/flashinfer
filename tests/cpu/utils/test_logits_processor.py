import pytest
import torch

from flashinfer.logits_processor import LogitsPipe, Softmax, Temperature


# compile=True would fuse Temperature -> Softmax into the CUDA softmax kernel.
@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab_size", [2, 3])
@pytest.mark.parametrize(
    "compile, with_softmax", [(False, False), (True, False), (False, True)]
)
def test_temperature_per_batch_tensor(dtype, vocab_size, compile, with_softmax):
    # With vocab_size == 2 (the batch size), scaling columns would not raise.
    logits = torch.arange(1, vocab_size + 1, dtype=dtype).repeat(2, 1)
    temperature = torch.tensor([0.25, 2.0], dtype=dtype)
    processors = [Temperature(), Softmax()] if with_softmax else [Temperature()]

    out = LogitsPipe(processors, compile=compile)(logits, temperature=temperature)

    expected = logits / temperature[:, None]
    if with_softmax:
        expected = torch.softmax(expected, dim=-1)
    torch.testing.assert_close(out, expected)


@pytest.mark.cpu_only
def test_temperature_tensor_length_mismatch_raises():
    # Treating it as per-row would broadcast the one row of logits into four.
    with pytest.raises(ValueError, match="TemperatureOp"):
        LogitsPipe([Temperature()])(torch.zeros(1, 3), temperature=torch.ones(4))
