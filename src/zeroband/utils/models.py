import os
from typing import Literal, TypeAlias

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM

ModelType: TypeAlias = LlamaForCausalLM | Qwen2ForCausalLM | Qwen3ForCausalLM
AttnImpl: TypeAlias = Literal["sdpa", "flash_attention_2"]
CONFIG_CACHE = f'./.config_cache'

def get_model_and_tokenizer(model_name: str, attn_impl: AttnImpl) -> tuple[ModelType, AutoTokenizer]:
    if os.path.exists(f'{CONFIG_CACHE}/{model_name}.json'):
        config_model = AutoConfig.from_pretrained(f'{CONFIG_CACHE}/{model_name}.json', attn_implementation=attn_impl)
    else:
        config_model = AutoConfig.from_pretrained(model_name, attn_implementation=attn_impl)
        parent_dir = os.path.dirname(f'{CONFIG_CACHE}/{model_name}')
        os.makedirs(parent_dir, exist_ok=True)
        print(f'Dumping config to {CONFIG_CACHE}/{model_name}.json')
        config_model.to_json_file(f'{CONFIG_CACHE}/{model_name}.json')

    config_model.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, config=config_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer 
