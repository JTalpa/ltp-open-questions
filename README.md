# OpenBART: Generating Open Questions
We present OpenBART, a Natural Language Processing model based on BART generating relevant open questions from input paragraphs. Made for a Language Technology Practical course.

## Model Use

The main way to use the model is through _generate.py_ to generate an open question from a user-inputted paragraph.
There are two ways to do this, both require the model folder _model-OpenBART_ to be present in the folder.

The first way to run _generate_ is through the command line, by executing the following command:
```console
python3 generate.py
```

The second way to run _generate_ is by importing it, as shown below:
```python
import generate

input_string = "This is an example input paragraph."
generate.generate_question(input_string)
```

It is possible to run these with the model present in a different folder. Additionally, using the _import_ method, one can specify whether data should be preprocessed:
```console
python3 generate.py path/to/folder_that_contains_model_folder
```
```python
folder = "path/to/folder_that_contains_model_folder
preprocess = True
generate.generate_question(input_string, folder, preprocess)
```

## Reproducing Results

In this section, we will describe the methods that led to the model & the evaluation scores, and how to reproduce them.
Firstly, to install all relevant packages and dependencies, download _requirements.txt_ and run the following:
```console
pip install -r requirements.txt
```
### Preprocessing Data

Preprocessing involves two files: _main.py_, which is executed, and _prepdata.py_, which is imported by _main.py_.
_main_ takes a single split (train, test, validation1 or validation2) from _rexarski/eli5-category_ on huggingface.co and preprocesses it by running it through an NER tagger and a Keyword Extractor.
```console
python3 main.py split (path/to/save_folder)
```

### Model Training

Model training is done using _train_model.py_. This program is more flexible and has more opportunity for customisation. It is called using the following:
```console
python3 train_model.py save_folder_name
```
and takes the following arguments:
```console
-m		--model					Destination name for checkpoints, results and final model
-t 		--tokenizer			Source name of tokenizer to use
-d 		--dataset				Source name of model to train
-p 		--path					Path to source & destination folders
-e 		--epochs				Number of epochs to train the model
-l 		--learningrate	Learning rate of the model
-b		--batchsize			Batch sizes of the model
-c 		--cpu						Use CPU instead of GPU
-q 		--checkpoint		Continue from specified checkpoint
```

To train the final model, this script was executed with the following parameters:
```console
python3 train_model.py "four-epochs" --epochs 4 --batchsize 8 --learningrate 2e-5
```
Parameters that were left out had default values instantiated by the script.
To reproduce these results on a different machine, the _path_ must be altered to fit the machine layout.

This code outputs a model (with the three most recent checkpoints and the best checkpoint) in `--path/save_folder_name`

### Model Evaluation

