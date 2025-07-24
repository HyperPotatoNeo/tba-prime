import pytest

from zeroband.inference.work_counting import get_inference_input_output_flops


@pytest.mark.parametrize(
    "model_name_or_path, active_params",
    [
        ("deepseek-ai/DeepSeek-R1-0528", 37e9),  # 37B Active Params from https://arxiv.org/pdf/2412.19437
    ],
)
def test_get_inference_input_output_flops_deepseek_v3(model_name_or_path: str, active_params: int):
    # 1 input token, 0 output tokens should be almost equal to 2 * active params
    input_flops, output_flops = get_inference_input_output_flops(model_name_or_path, 1, 1)
    assert abs(input_flops / 8 - 2 * active_params) / active_params < 0.05
    assert output_flops > input_flops


@pytest.mark.parametrize(
    "model_name_or_path, active_params",
    [
        ("Qwen/Qwen3-0.6B", 0.6e9),
        ("Qwen/Qwen3-1.7B", 1.7e9),
        ("Qwen/Qwen3-4B", 4e9),
        ("Qwen/Qwen3-8B", 7.6e9),  # This is only 8B because it has untied embs somehow
        ("Qwen/Qwen3-14B", 14e9),
        ("Qwen/Qwen3-32B", 32e9),
        ("Qwen/Qwen3-30B-A3B", 3e9),
        ("Qwen/Qwen3-235B-A22B", 22e9),
    ],
)
def test_get_inference_input_output_flops_qwen3(model_name_or_path: str, active_params: int):
    input_flops, output_flops = get_inference_input_output_flops(model_name_or_path, 1, 1)
    assert abs(input_flops - 2 * active_params) / active_params < 0.05
    assert output_flops > input_flops


@pytest.mark.parametrize(
    "model_name_or_path, active_params",
    [
        ("PrimeIntellect/INTELLECT-1", 9.5e9),  # Decoupled embs of ~5e8 params
        ("Qwen/Qwen2.5-0.5B", 0.5e9),  # Tied embs
    ],
)
def test_get_inference_input_output_flops_fallback(model_name_or_path: str, active_params: int):
    input_flops, output_flops = get_inference_input_output_flops(model_name_or_path, 1, 1)
    assert abs(input_flops - 2 * active_params) / active_params < 0.05
    assert output_flops == input_flops
