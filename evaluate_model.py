import sys
from pathlib import Path
from evaluate import load
from datasets import load_from_disk
from generate import generate_question

from time import time
import numpy as np
import bert_score

folder = Path('/scratch/s4641353/ltp')

# load dataset
print('loading dataset')
dataset = load_from_disk((folder / 'dataset-prepped2/validation2').as_posix()).to_iterable_dataset().take(100)


if __name__ == "__main__":

    model_label = sys.argv[1]

    now = time()

    print("loading BERTScore")
    bertscore = load("bertscore", model_name="microsoft/deberta-xlarge-mnli")
    print("loading BLEURT")
    bleurt = load("bleurt", module_type="metric", checkpoint="bleurt-base-512")
#    print("loading BLEU")
#    bleu = load("bleu")
#    print("loading ROUGE")
#    rouge = load("rouge", rouge_types=['rouge1', 'rouge2', 'rougeL'])

    print('generating questions')
    inputs = [a['answers']['text'][0] for a in dataset]
    predictions = [generate_question(a['answers']['text'][0], preprocess_string=False) for a in dataset]
    references = [q['title'] for q in dataset]

    print("computing BERTScore")
    bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en",
                                      rescale_with_baseline=True)
    
    bertscore_2 = bert_score.score(predictions, references, model_type="microsoft/deberta-large-mnli")
    # pip install bert_score
    print("computing BLEURT")
    bleurt_res = bleurt.compute(predictions=predictions, references=references)
    # pip install git+https://github.com/google-research/bleurt.git
    # print("computing BLEU")
    # bleu_res = bleu.compute(predictions=predictions, references=references)
    # print('computing ROUGE')
    # rouge_res = rouge.compute(predictions=predictions, references=references)
    # # pip install rouge_score

    hashcode = bertscore_res.pop('hashcode')
    mean_bertscores = {key: np.mean(val) for key, val in bertscore_res.items()}
    
    bleurts = bleurt_res['scores']

    print(f"BERTScore Metrics: {mean_bertscores}")
    print(f"BERTScore Hashcode: {hashcode}\n")
    print(f"BLEURT Score: {np.mean(bleurts)}")
    # print(f"BLEU Score: {bleu_res['bleu']}")
    # print(f"ROUGE Score: (ROUGE-1 | {rouge_res['rouge1']}) (ROUGE-2 | {rouge_res['rouge2']}) (ROUGE-L | {rouge_res['rougeL']})")
    print(f"Time elapsed: {time() - now} seconds")

    print(inputs[0], ' -> ', predictions[0])
