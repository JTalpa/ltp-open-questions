import argparse
import os
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    TrainerCallback
import sys
from datasets import load_from_disk
from time import time
from pathlib import Path


class PrintLossesCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        print(
            f"Epoch {state.epoch}: Training loss: {logs.get('loss', 'N/A')}, Validation loss: {logs.get('eval_loss', 'N/A')}")
        with open(folder / f'model-{output_loc}' / 'dat' / 'losses.log', 'a') as file:
            file.write(
                f"{state.epoch}, {state.global_step}, {logs.get('loss', 'N/A')}, {logs.get('eval_loss', 'N/A')}\n")


# ARGUMENTS: output model name; learning rate; batch size; epochs; cpu
if __name__ == '__main__':
    print("Python: Start")

    default_epochs = 4
    default_learningrate = float(2e-5)
    default_batchsize = 4

    default_model = 'bart'
    default_tokenizer = 'bart'
    default_dataset = 'tokenized'
    default_path = Path('/scratch/s4641353/ltp/')

    parser = argparse.ArgumentParser(prog="train_model.py",
                                     description="Train a BART model on the ELI5-category dataset")

    parser.add_argument('destination', help='Destination name for checkpoints, results and final model')
    parser.add_argument('-m', '--model', type=str, default=default_model, help='Source name of model to train')
    parser.add_argument('-t', '--tokenizer', type=str, default=default_tokenizer,
                        help='Source name of tokenizer to use')
    parser.add_argument('-d', '--dataset', type=str, default=default_dataset, help='Source name of model to train')
    parser.add_argument('-p', '--path', type=Path, default=default_path, help="Path to source & destination folders")

    parser.add_argument('-e', '--epochs', type=int, default=default_epochs, help='Number of epochs to train the model')
    parser.add_argument('-l', '--learningrate', type=float, default=default_learningrate,
                        help='Learning rate of the model')
    parser.add_argument('-b', '--batchsize', type=int, default=default_batchsize, help='Batch sizes of the model')
    parser.add_argument('-c', '--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('-q', '--checkpoint', type=Path, default=False, help='Continue from specified checkpoint')

    args = parser.parse_args()

    output_loc = args.destination

    folder = args.path
    # mp.set_start_method('spawn')
    # poolSize = int(os.environ['SLURM_GPUS_ON_NODE']) if os.environ['SLURM_GPUS_ON_NODE'] else 1

    begin_time = time()
    print('loading model')
    model = BartForConditionalGeneration.from_pretrained(folder / f'model-{args.model}')
    print('loading tokenizer')
    bart_tokenizer = BartTokenizer.from_pretrained(folder / f'tokenizer-{args.tokenizer}')
    print('loading dataset')
    ds = load_from_disk((folder / f'dataset-{args.dataset}').as_posix())

    (folder / f'model-{output_loc}' / 'dat').mkdir(parents=True, exist_ok=True)
    with open(folder / f'model-{output_loc}' / 'dat' / 'losses.log', 'a') as f:
        f.write('epoch, step, training_loss, eval_loss\n')

    training_args = Seq2SeqTrainingArguments(
        # torch_compile=True,
        output_dir=(folder / f'model-{output_loc}' / 'dat').as_posix(),

        logging_strategy="steps",
        eval_strategy="steps",
        logging_steps=500,

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=4,
        load_best_model_at_end=True,

        learning_rate=args.learningrate,
        per_device_train_batch_size=args.batchsize,
        per_device_eval_batch_size=args.batchsize,
        num_train_epochs=args.epochs,
        weight_decay=0.01,

        resume_from_checkpoint=args.checkpoint,

        use_cpu=args.cpu,
        log_level='info',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        callbacks=[PrintLossesCallback],
    )

    print('fine-tuning model')
    trainer.train()

    model.save_pretrained(folder / f'model-{output_loc}')
