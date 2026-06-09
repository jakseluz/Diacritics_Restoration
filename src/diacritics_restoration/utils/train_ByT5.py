from __future__ import annotations

import argparse
import os
import random

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from diacritics_restoration.utils.dataset import get_wikipedia_data


def make_translate_table():
    mapping = {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
        "Ą": "A",
        "Ć": "C",
        "Ę": "E",
        "Ł": "L",
        "Ń": "N",
        "Ó": "O",
        "Ś": "S",
        "Ź": "Z",
        "Ż": "Z",
    }
    return str.maketrans(mapping)


def remove_diacritics(text: str, translate_table: dict) -> str:
    return text.translate(translate_table)


def main():
    ap = argparse.ArgumentParser(
        description="Train ByT5 model for diacritics restoration"
    )
    ap.add_argument(
        "--model_name",
        type=str,
        default="google/byt5-small",
        help="Pre-trained ByT5 model name",
    )
    ap.add_argument(
        "--num_articles",
        type=int,
        default=10000,
        help="Number of Wikipedia articles to use for training",
    )
    ap.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="./models/ByT5_Diacritics",
        help="Directory to save the trained model",
    )
    ap.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    ap.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training",
    )
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size per device during evaluation",
    )
    ap.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for training",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    translate_table = make_translate_table()

    df = get_wikipedia_data(num_articles=args.num_articles)
    texts = df["text"].astype(str).tolist()

    random.shuffle(texts)

    n = len(texts)
    n_validation = max(1, int(0.1 * n))
    validation_texts = texts[:n_validation]
    train_texts = texts[n_validation:]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": validation_texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model.config.tie_word_embeddings = False
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def preprocess(batch):
        targets = batch["text"]
        inputs = [remove_diacritics(t, translate_table) for t in targets]

        model_inputs = tokenizer(
            inputs,
            truncation=True,
            padding=False,
            max_length=args.max_input_length,
        )

        labels = tokenizer(
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=args.max_input_length,
        )

        label_ids = labels["input_ids"]
        pad_id = tokenizer.pad_token_id
        label_ids = [
            [(token if token != pad_id else -100) for token in seq]
            for seq in label_ids
        ]

        model_inputs["labels"] = label_ids
        return model_inputs

    train_tokenized = train_ds.map(
        preprocess, batched=True, remove_columns=["text"]
    )
    val_tokenized = val_ds.map(
        preprocess, batched=True, remove_columns=["text"]
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # training
    os.makedirs(args.output_dir, exist_ok=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        predict_with_generate=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        optim="adafactor",
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    main()
