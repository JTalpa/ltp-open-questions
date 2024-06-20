from transformers import BartTokenizer
import sys
import os
from functools import partial

from datasets import load_from_disk, DatasetDict
from time import time

from prepdata import preprocess
import multiprocessing as mp


def tokenize(text, tokenizer):
    model_inputs = tokenizer(text=text['answer'], text_target=text['question'],
                             max_length=1024, truncation=True, padding='max_length')
    return model_inputs


def filter_data(data):
    length_answer = sum(data['attention_mask'])
    length_question = sum([x != 1 for x in data['labels']])

    return 15 < length_answer <= 350 and 8 < length_question <= 35


if __name__ == '__main__':
    folder = '/scratch/s4641353/ltp'

    split = sys.argv[1]
    print("Python: Start")
    mp.set_start_method('spawn')
    poolSize = int(os.environ['SLURM_GPUS_ON_NODE']) if os.environ['SLURM_GPUS_ON_NODE'] else 1

    print('loading tokenizer')
    bart_tokenizer = BartTokenizer.from_pretrained('/scratch/s4641353/ltp/tokenizer-bart')
    print('loading dataset')
    dataset_full = load_from_disk('/scratch/s4641353/ltp/dataset-eli5')

    ds = dataset_full[split]
    ds.shuffle()

    ds_prepped = ds.map(preprocess, num_proc=poolSize)

    ds_prepped.save_to_disk(f'{folder}/dataset-prepped2/{split}')

    ds_tokenized = ds_prepped.map(partial(tokenize, tokenizer=bart_tokenizer))

    size_total = len(list(ds_tokenized))

    ds_filtered = ds_tokenized.filter(filter_data)

    size_filtered = len(list(ds_filtered))

    ds_filtered.save_to_disk(f'{folder}/dataset-tokenized/{split}')

    print(f"Filtered {size_total} -> {size_filtered}")
