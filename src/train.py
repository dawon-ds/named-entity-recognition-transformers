import os

import matplotlib.pyplot as plt
from transformers import (
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from src.metrics import compute_metrics


def train(model, datasets, tokenizer, config, label_list):
    """Fine-tune a token-classification model and save training plots."""
    os.makedirs(config["output_dir"], exist_ok=True)

    data_collator = DataCollatorForTokenClassification(
        tokenizer,
        padding=True,
    )
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(config["learning_rate"]),
        per_device_train_batch_size=int(config["batch_size"]),
        per_device_eval_batch_size=int(config["batch_size"]),
        num_train_epochs=float(config["num_epochs"]),
        weight_decay=float(config["weight_decay"]),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        report_to=[],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda prediction: compute_metrics(
            prediction,
            label_list,
        ),
    )
    trainer.train()
    save_training_plots(trainer.state.log_history, config["output_dir"])
    return trainer


def save_training_plots(history, output_dir):
    """Save loss and validation F1 curves from Trainer logs."""
    train_loss = [item["loss"] for item in history if "loss" in item]
    eval_loss = [
        item["eval_loss"] for item in history if "eval_loss" in item
    ]
    eval_f1 = [item["eval_f1"] for item in history if "eval_f1" in item]

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    if eval_f1:
        plt.figure()
        plt.plot(eval_f1, marker="o", label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "f1_plot.png"))
        plt.close()
