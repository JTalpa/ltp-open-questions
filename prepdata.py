import flair
from time import time
import re
import yake

print('loading ner tagger')
tagger = flair.models.SequenceTagger.load("flair/ner-english")


# INPUT: sentence to tag (Sentence), term list (dict { MISC (list), PER (list), ORG (list), LOC (list) })
# OUTPUT: tagged sentence (str), updated term list (dict { MISC (list), PER (list), ORG (list), LOC (list) })
def replace_entities(sentence, term_list):
    string = sentence.to_plain_string()
    entities = sentence.to_dict()['entities']

    for entity in entities:
        term = entity['text']
        label = entity['labels'][0]['value']

        if term in term_list[label]:
            mask = f"{label}{term_list[label].index(term)}"
        else:
            mask = f"{label}{len(term_list[label])}"
            term_list[label].append(term)

        string = re.sub(fr"\b{re.escape(term)}\b", mask, string, 1)

    return string, term_list


# INPUT: ELI5_category-formatted data
# OUTPUT: dict(answer (str), question (str),
#         term list (dict { MISC (list), PER (list), ORG (list), LOC (list) }, time spent (list)) )
def ner(data):
    answer = flair.data.Sentence(data['answers']['text'][0])
    question = flair.data.Sentence(data['title'])

    start = time()
    tagger.predict(answer)
    tagger.predict(question)
    tagging_time = time() - start

    term_list = {'MISC': [],
                 'PER': [],
                 'ORG': [],
                 'LOC': [],
                 }

    str_answer, term_list = replace_entities(answer, term_list)

    str_question, term_list = replace_entities(question, term_list)

    total_time = time() - start
    return dict({'answer': str_answer, 'question': str_question, 'labels': term_list, 'time': [tagging_time, total_time]})


def ner_gen(data):
    answer = flair.data.Sentence(data)

    start = time()
    tagger.predict(answer)
    tagging_time = time() - start

    term_list = {'MISC': [],
                 'PER': [],
                 'ORG': [],
                 'LOC': [],
                 }

    str_answer, term_list = replace_entities(answer, term_list)

    total_time = time() - start
    return dict({'answer': str_answer, 'labels': term_list, 'time': [tagging_time, total_time]})


# INPUT: text (str)
# OUTPUT: keywords (list)
def get_keywords(text):
    language = "en"
    max_ngram_size = 1
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    window_size = 1
    num_of_keywords = 3

    kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                         dedupFunc=deduplication_algo, windowsSize=window_size, top=num_of_keywords,
                                         features=None)

    kw_data = kw_extractor.extract_keywords(text)
    keywords = [kw[0] for kw in kw_data]
    return keywords


# INPUT: ELI5_category-formatted data
# OUTPUT: dict(answer (str), question (str),
#         term list (dict { MISC (list), PER (list), ORG (list), LOC (list) }, time spent (list)) )
def preprocess(data, **kwargs):

    if 'mode' in kwargs and kwargs['mode'] == 'generate':
        ner_data = ner_gen(data)
    else:
        ner_data = ner(data)

    answer = ner_data['answer']
    kw_answer = ' '.join(get_keywords(answer)) + ' ' + answer
    ner_data.update({'answer': kw_answer})

    return ner_data

