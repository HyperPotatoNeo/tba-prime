from datasets import load_dataset
import numpy as np
import re
from zeroband.tasks.math_utils import compute_math_reward

class Dataset:
    def __init__(self, tokenizer, enable_thinking, chat_format=False):
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.chat_format = chat_format
        self.columns = ['prompt', 'ground_truth', 'reasoning']
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].rename_columns({"question": "prompt", "answer": "ground_truth"})
            self.dataset[split] = self.dataset[split].map(self.format_dataset, remove_columns=[col for col in self.dataset[split].column_names if col not in self.columns])

    def format_prompts(self, prompts, answers = None):
        prompts = [prompt + f" Place your answer at end exactly inside latex box for example \\boxed{{12}} or \\boxed{{51}}." for prompt in prompts]
        if self.chat_format:
            messages = [
                [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                for prompt in prompts
            ]
            if answers is not None:
                for msg, answer in zip(messages, answers):
                    msg.append(
                        {
                            "role": "assistant",
                            "content": f"The final answer is {answer}"
                        }
                    )
                add_generation_prompt = False
            else:
                add_generation_prompt = True

            # Apply chat template
            formatted_prompts = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=self.enable_thinking
            )
        else:
            if answers is not None:
                prompts = [f'{prompt} The answer is {answer} {self.tokenizer.eos_token}' for prompt, answer in zip(prompts, answers)]

            formatted_prompts = [self.tokenizer(prompt)["input_ids"] for prompt in prompts]

        return formatted_prompts
    
    def reward_fn(self, completion, ground_truth):
        return compute_math_reward(completion, ground_truth)

class GSM8k(Dataset):
    def __init__(self, tokenizer, enable_thinking, chat_format=False):
        self.dataset = load_dataset("openai/gsm8k", "main")

        super().__init__(tokenizer, enable_thinking, chat_format)

    def format_dataset(self, example):
        gt = example["ground_truth"].split('####')[-1].strip()
        reasoning = example["ground_truth"].split('####')[0]
        reasoning = re.sub(r"<<.*?>>", "", reasoning)
        example["ground_truth"] = f"\\boxed{{{gt}}}"
        example["reasoning"] = reasoning

        return example

    def extract_gt(self, solution):
        return solution.split('####')[-1].strip()

class MATH500(Dataset):
    def __init__(self, tokenizer, enable_thinking, chat_format = False, split: bool = False):
        self.dataset = load_dataset("HuggingFaceH4/MATH-500")
        if split:
            indices = np.arange(len(self.dataset['test']))
            indices = np.random.permutation(indices)
            num_test = len(indices) // 4
            self.dataset['train'] = self.dataset['test'].select(indices[:-num_test])
            self.dataset['test'] = self.dataset['test'].select(indices[-num_test:])

        super().__init__(tokenizer, enable_thinking, chat_format)

    def format_dataset(self, example):
        example['ground_truth'] = f"\\boxed{{{example['ground_truth']}}}"
        example["reasoning"] = example['solution']
        return example

class AIME:
    def __init__(self):
        pass

if __name__ == '__main__':
    dataset = MATH500()
    print(dataset('train')[0])