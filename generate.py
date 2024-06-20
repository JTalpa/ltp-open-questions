import sys
from pathlib import Path

from transformers import pipeline
from prepdata import preprocess
from transformers import BartTokenizer

import re

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')


def generate_question(in_str, folder="/scratch/s4641353/ltp", preprocess_string=True):

    generate = pipeline(task="text2text-generation", model=(Path(folder) / f"model-four-epochs").as_posix(), tokenizer=bart_tokenizer, max_new_tokens=512)

    if preprocess_string:
        in_data = preprocess(in_str, mode='generate')['answer']
    else:
        in_data = in_str

    generated = generate(in_data)
    print("A")
    if preprocess_string:
        out = unify_named_entities(generated[0]['generated_text'], in_data['labels'])
    else:
        out = generated[0]['generated_text']

    return out


def unify_named_entities(text, labels):

    for label, words in labels.items():
        if words:
            for i, word in enumerate(words):
                text = re.sub(fr"{label}{i}", word, text)
    return text


if __name__ == "__main__":

    folder_loc = sys.argv[1]

    another_question = 'y'
    while another_question == 'y':
        input_string = input("Enter Input Paragraph: ")
        question = generate_question(input_string, folder_loc, preprocess_string=True)
        print(f"Generated question: \"{question}\"")
        another_question = input("Generate another question? (y/n): ")


