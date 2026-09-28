import argparse

from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.data import load_conll2003, tokenize_and_align_labels
from src.train import train
from src.utils import get_device, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to an experiment YAML configuration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    device = get_device(config["use_gpu"])

    dataset = load_conll2003()
    label_list = dataset["train"].features["ner_tags"].feature.names

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        add_prefix_space=True,
    )
    tokenized_dataset = tokenize_and_align_labels(
        dataset,
        tokenizer,
        config["max_length"],
    )

    model = AutoModelForTokenClassification.from_pretrained(
        config["model_name"],
        num_labels=len(label_list),
    ).to(device)

    trainer = train(
        model,
        tokenized_dataset,
        tokenizer,
        config,
        label_list,
    )
    metrics = trainer.evaluate()
    print(f"Validation F1: {metrics['eval_f1']:.4f}")


if __name__ == "__main__":
    main()
