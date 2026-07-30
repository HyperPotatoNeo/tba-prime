from prime_rl.orchestrator.vf_utils import get_seq_len


def test_get_seq_len_uses_markovian_effective_prompt_tokens():
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4, 5],
                },
                "extras": {"effective_prompt_tokens": 10},
            }
        ]
    }

    assert get_seq_len(output) == 13


def test_get_seq_len_preserves_physical_default():
    output = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "completion_ids": [3, 4, 5],
                },
                "extras": {},
            }
        ]
    }

    assert get_seq_len(output) == 5
