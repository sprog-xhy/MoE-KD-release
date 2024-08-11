import json
import os.path
import pickle
import time

import datasets
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def read_json(json_path: str):
    if json_path.endswith('.json'):
        with open(json_path, 'r') as f:
            return json.load(f)
    elif json_path.endswith('.jsonl'):
        with open(json_path, 'r') as f:
            data = []
            for line in f:
                data.append(json.loads(line))
            return data
    else:
        raise ValueError()


def get_prompt(instruction: str, intput: str, output: str, eos_token, template):
    if template == 1:
        # IT
        if len(intput) > 0:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n"
            )
            prompt = template.format(instruction, intput)
        else:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{}\n\n### Response:\n"
            )
            prompt = template.format(instruction)
    elif template == 2:
        # TS
        template = (
            "Summarize the following conversation\n\n### Conversation:\n{}\n\n### Summary:\n"
        )
        prompt = template.format(instruction)
    elif template == 3:
        # QA
        template = (
            "### Question:\n{}\n\n### Answer:\n"
        )
        prompt = template.format(instruction)
    elif template == 4:
        # MT
        template = (
            "Translate the following English into Chinese\n\n### English:\n{}\n\n### Chinese:\n"
        )
        prompt = template.format(instruction)
    else:
        raise NotImplementedError(f"Check your template id.")

    output = output + " " + eos_token
    prompt_resp = prompt + output  # end of conversation
    return prompt_resp, prompt, output


class PromptDataset(Dataset):
    def __init__(self, input_ids: list, attention_mask: list, labels: list, prompt_ids: list, prompt_mask: list, resp_ids: list, train_or_eval: str):
        # train
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.train_or_eval = train_or_eval
        if train_or_eval == "eval":
            # generation and evaluation
            self.prompt_ids = prompt_ids
            self.prompt_mask = prompt_mask
            self.resp_ids = resp_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.train_or_eval == 'train':
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],
            }
        elif self.train_or_eval == 'eval':
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx],
                'prompt_ids': self.prompt_ids[idx],
                'prompt_mask': self.prompt_mask[idx],
                'resp_ids': self.resp_ids[idx]
            }


def get_labels(input_ids, attention_mask, prompt_mask):
    assert input_ids.shape == attention_mask.shape == prompt_mask.shape
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    labels[prompt_mask == 1] = -100
    return labels


def tokenize_prompts(data, tokenizer, max_seq_len, train_or_eval: str, only_resp_loss, template):
    if data == []:
        return ([] for _ in range(6))
    assert not any([x not in data[0].keys() for x in ['instruction', 'input', 'output']])
    assert train_or_eval in ['train', 'eval']
    input_ids, attention_mask, labels = [], [], []  # for train, right pad
    prompt_ids, prompt_mask, resp_ids = [], [], []  # for eval , left pad
    max_prompt_len = max_seq_len // 2 if template != 4 else max_seq_len - 64
    for d in tqdm(data):
        prompt_resp, prompt, resp = get_prompt(d['instruction'], d['input'], d['output'], tokenizer.eos_token, template)
        embedding = tokenizer(prompt_resp, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True)
        input_ids.append(embedding.input_ids[0])
        attention_mask.append(embedding.attention_mask[0])
        # for labels
        if only_resp_loss:
            right_pad_prompt_mask = tokenizer(prompt, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True).attention_mask
            labels.append(get_labels(embedding.input_ids, embedding.attention_mask, right_pad_prompt_mask)[0])
        else:
            labels.append(embedding.input_ids[0].clone())
        if train_or_eval == 'eval':
            tokenizer.padding_side = 'left'
            prompt_embedding = tokenizer(prompt, return_tensors='pt', max_length=max_prompt_len, padding='max_length', truncation=True)
            prompt_ids.append(prompt_embedding.input_ids[0])
            prompt_mask.append(prompt_embedding.attention_mask[0])
            resp_ids.append(tokenizer(resp, return_tensors='pt', max_length=max_seq_len, padding='max_length', truncation=True).input_ids[0])
            tokenizer.padding_side = 'right'
    return input_ids, attention_mask, labels, prompt_ids, prompt_mask, resp_ids


def create_sft_dataset(data_path: str, max_seq_len, tokenizer, cache_path, eval_num: int, only_resp_loss, template, reload=False):
    beg_time = time.time()
    print("---------------------------------------------")
    print("|---> Creating sft_dataset ...")
    if cache_path is None:
        cache_path = os.path.join(data_path, "cached_data/")
    if reload or not os.path.exists(cache_path):
        suffix = '.json'
        if not any([x.endswith(suffix) for x in os.listdir(data_path)]):
            suffix = ".jsonl"
        train_data = read_json(os.path.join(data_path, f'train{suffix}')) if os.path.exists(os.path.join(data_path, f'train{suffix}')) else []  # [:256]
        eval_data = read_json(os.path.join(data_path, f'eval{suffix}'))[:eval_num]
        train_embeddings = tokenize_prompts(train_data, tokenizer, max_seq_len, 'train', only_resp_loss, template)
        eval_embeddings = tokenize_prompts(eval_data, tokenizer, max_seq_len, 'eval', only_resp_loss, template)
        train_dataset = PromptDataset(*train_embeddings, train_or_eval='train')
        eval_dataset = PromptDataset(*eval_embeddings, train_or_eval='eval')
        save_pickle(cache_path, train_dataset, eval_dataset)
    else:
        train_dataset, eval_dataset = load_pickle(cache_path)
    print(f"|---> Cost time: {time.time() - beg_time:.2f}s")
    print("---------------------------------------------")
    return train_dataset, eval_dataset


def save_pickle(cache_path, train_dataset: PromptDataset, eval_dataset: PromptDataset):
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    with open(os.path.join(cache_path, 'train.pickle'), 'wb') as f:
        pickle.dump(train_dataset, f)
    with open(os.path.join(cache_path, 'eval.pickle'), 'wb') as f:
        pickle.dump(eval_dataset, f)


def load_pickle(cache_path):
    with open(os.path.join(cache_path, 'train.pickle'), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(cache_path, 'eval.pickle'), 'rb') as f:
        eval_dataset = pickle.load(f)
    return train_dataset, eval_dataset
