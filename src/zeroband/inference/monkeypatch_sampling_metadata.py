from typing import Optional

import torch
from vllm.model_executor.sampling_metadata import SamplingMetadataCache, SequenceGroupToSample
from vllm.sampling_params import SamplingType
from vllm.sequence import SequenceGroupMetadata


def fixed_prepare_seq_groups(
    seq_group_metadata_list: list[SequenceGroupMetadata],
    seq_lens: list[int],
    query_lens: list[int],
    device: str,
    generators: Optional[dict[str, torch.Generator]] = None,
    cache: Optional[SamplingMetadataCache] = None,
) -> tuple[
    list[SequenceGroupToSample],
    list[int],
    dict[SamplingType, list[int]],
    int,
]:
    """Prepare sequence groups and indices for sampling.

    Args:
        seq_group_metadata_list: A list of sequence group to batch.
        seq_lens: A list of sequence lens per sequence group.
            Index of prompt len should match with seq_group_metadata_list.
        query_lens: A list of query lengths. Prompt lens include the length
            of entire prompt tokens, and it could be shorter.
        device: A device to use for random number generators,
            `SequenceGroupToSample.generator`.
        generators: A store of per-request random number generators used
            for seeded requests.

    Returns:
        seq_groups: A list of sequence group to sample.
        selected_token_indices: See the definition from `SamplingMetadata`.
        categorized_sample_indices: See the definition from `SamplingMetadata`.
        num_prompts: Total number of prompts from `seq_group_metadata_list`.
    """
    # Batched sequence groups for the current model forward stsep.
    seq_groups: list[SequenceGroupToSample] = []
    # A list of token indices to sample/compute logprob. It is used to
    # prune the outcome logits from the model for the performance.
    selected_token_indices: list[int] = []
    # Used for selected_token_indices.
    model_output_idx = 0

    # Sampling type -> (
    # indices to sample/prompt logprob within pruned output logits,
    # indices to sample within pruned logits)
    categorized_sample_indices: dict[SamplingType, list[int]] = {t: [] for t in SamplingType}
    # Index of logits to compute logprob. Logits include both prompt logprob
    # and sample logprob indices.
    logit_idx = 0
    # Total number of prompts from given sequence groups.
    num_prompts = 0

    for i, seq_group_metadata in enumerate(seq_group_metadata_list):
        seq_ids = seq_group_metadata.seq_data.keys()

        if cache is not None:
            sample_obj = cache.get_cached_seq_group_to_sample(len(seq_ids))

            for j, seq_id in enumerate(seq_ids):
                sample_obj.seq_ids[j] = seq_id

            sample_obj.prompt_logprob_indices.clear()
            sample_obj.sample_indices.clear()

        sampling_params = seq_group_metadata.sampling_params
        is_prompt = seq_group_metadata.is_prompt
        generator: Optional[torch.Generator] = None
        # If the current seq group is in decode stage, it is None.
        seq_len: Optional[int] = None
        query_len: Optional[int] = None
        prompt_logprob_indices: list[int] = sample_obj.prompt_logprob_indices if cache is not None else []
        sample_indices: list[int] = sample_obj.sample_indices if cache is not None else []
        do_sample = seq_group_metadata.do_sample

        if seq_group_metadata.is_prompt:
            # This is the fix
            if seq_group_metadata.request_id in generators:
                generator = generators[seq_group_metadata.request_id]
            elif sampling_params.seed is not None:
                generator = torch.Generator(device=device).manual_seed(sampling_params.seed)
                if generators is not None:
                    generators[seq_group_metadata.request_id] = generator

            num_prompts += 1
            num_prefill_sample = len(seq_ids)
            assert num_prefill_sample == 1
            assert query_lens is not None and seq_lens is not None
            query_len, seq_len = query_lens[i], seq_lens[i]
            # If we need sampling, exclude num_prefill_sample tokens from
            # prompt logprob.
            prompt_logprob_len = query_len - num_prefill_sample if do_sample else query_len
            sample_len = num_prefill_sample if do_sample else 0
        else:
            # Decode
            prompt_logprob_len = 0
            query_len = query_lens[i] if query_lens is not None and len(query_lens) > 0 else 1
            sample_len = len(seq_ids) * query_len if do_sample else 0

            if sampling_params.seed is not None and generators is not None:
                generator = generators.get(seq_group_metadata.request_id)

        # Update indices to select from the model output.
        """
        This blocks computes selected_token_indices which is used in the
        following way.

        hidden_states = model(...)
        logits = hidden_states[selected_token_indices]
        """

        if sampling_params.prompt_logprobs is not None:
            selected_token_indices.extend(range(model_output_idx, model_output_idx + prompt_logprob_len))
        model_output_idx += prompt_logprob_len
        if do_sample:
            selected_token_indices.extend(range(model_output_idx, model_output_idx + sample_len))
        model_output_idx += sample_len

        # We now find indices for logprob computation and sampling.
        """
        This block computes categorized_sample_indices which is used in the
        following way.

        hidden_states = model(...)
        logits = hidden_states[selected_token_indices]
        def sample(logits):
           # Use categorized_sample_indices for sampling.
           # prompt_logprob_indices to find prompt logprob indices.
           # sample_indices to find sample indices.
        """

        if sampling_params.prompt_logprobs is not None:
            prompt_logprob_indices.extend(range(logit_idx, logit_idx + prompt_logprob_len))
            logit_idx += prompt_logprob_len
        if do_sample:
            sample_indices.extend(range(logit_idx, logit_idx + sample_len))
            categorized_sample_indices[sampling_params.sampling_type].extend(list(range(logit_idx, logit_idx + sample_len)))
            logit_idx += sample_len

        if cache is not None:
            sample_obj.sampling_params = sampling_params
            sample_obj.seq_data = seq_group_metadata.seq_data
            sample_obj.seq_len = seq_len
            sample_obj.query_len = query_len
            sample_obj.generator = generator
            sample_obj.is_prompt = is_prompt
        else:
            sample_obj = SequenceGroupToSample(
                seq_ids=list(seq_ids),
                sampling_params=sampling_params,
                seq_data=seq_group_metadata.seq_data,
                seq_len=seq_len,
                query_len=query_len,
                generator=generator,
                is_prompt=is_prompt,
                prompt_logprob_indices=list(prompt_logprob_indices),
                sample_indices=list(sample_indices),
            )

        seq_groups.append(sample_obj)

    if cache is not None:
        cache.reset()

    return (seq_groups, selected_token_indices, categorized_sample_indices, num_prompts)


from vllm.model_executor import sampling_metadata

sampling_metadata._prepare_seq_groups = fixed_prepare_seq_groups
