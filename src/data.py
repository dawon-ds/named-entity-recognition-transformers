from datasets import load_dataset


def load_conll2003():
    """Load the CoNLL-2003 named entity recognition dataset."""
    return load_dataset("conll2003")


def tokenize_and_align_labels(dataset, tokenizer, max_length):
    """Align word-level NER labels with tokenizer subword outputs."""

    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=max_length,
        )

        aligned_labels = []
        for batch_index, labels in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            label_ids = []

            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != previous_word_id:
                    label_ids.append(labels[word_id])
                else:
                    label_ids.append(-100)
                previous_word_id = word_id

            aligned_labels.append(label_ids)

        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset
