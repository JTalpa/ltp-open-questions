from transformers import BartForConditionalGeneration, BartTokenizer
from pathlib import Path

if __name__ == "__main__":

    print("loading resources")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    path = Path("./")

    print("saving resources")
    tokenizer.save_pretrained(path / 'tokenizer-bart')
    model.save_pretrained(path / 'model-bart')
