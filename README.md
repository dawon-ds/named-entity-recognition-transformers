# Named Entity Recognition with RoBERTa & ALBERT

A token-classification project comparing **RoBERTa-base** and **ALBERT-base-v2** on the **CoNLL-2003 Named Entity Recognition (NER)** dataset.

## Project Overview

The project fine-tunes pretrained Transformer models under the same training setup and compares their validation F1 scores.

| Model | Validation F1 |
| --- | ---: |
| RoBERTa-base | **0.9588** |
| ALBERT-base-v2 | **0.9394** |

Both experiments used 3 epochs, batch size 16, learning rate 2e-5, and weight decay 0.01. The configuration files also specify maximum sequence length 256 and seed 42.

The scores are historical validation results from the submitted assignment evaluation sheet, rather than results reproduced in this documentation update. RoBERTa's reported F1 is higher by **0.0194**, or **1.94 percentage points**.

## Implementation

- Loaded the CoNLL-2003 NER dataset with Hugging Face Datasets.
- Tokenized pre-split words with a pretrained tokenizer.
- Aligned word-level NER labels to subword tokens.
- Ignored special tokens and repeated subword pieces with label `-100`.
- Fine-tuned Transformer token-classification models with Hugging Face Trainer.
- Evaluated precision, recall, and F1 using `seqeval`.
- Compared RoBERTa-base and ALBERT-base-v2 under the same training configuration.

## Label Alignment & Evaluation

CoNLL-2003 supplies labels for words, but Transformer tokenizers may split each word into several pieces. The implementation uses `word_ids()` to map subwords back to the original words:

| Token position | Assigned label |
| --- | --- |
| First subword of a word | Original word-level NER label |
| Later subwords of the same word | `-100` |
| Special and padding tokens | `-100` |

Masked positions are ignored by the training loss and removed before metric computation. Predictions are converted back to label sequences and scored with `seqeval`, which measures entity-level precision, recall, and F1.

Training uses the dataset's training split, and evaluation uses its validation split. The best checkpoint is selected by validation F1 and restored at the end of training. The current entry point reports validation F1; it does not evaluate the test split.

## Project Structure

```text
.
├── config/
│   ├── roberta.yaml
│   └── albert.yaml
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── train.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Run

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Run either configuration:

```bash
python main.py --config config/roberta.yaml
python main.py --config config/albert.yaml
```

The program loads CoNLL-2003 and pretrained model/tokenizer files through Hugging Face. First execution requires access to these resources or an existing local cache.

Checkpoints and plots are written under `outputs/roberta/` or `outputs/albert/`:

- `checkpoint-*/` — saved model checkpoints
- `loss_plot.png` — training and validation loss
- `f1_plot.png` — validation F1

## Limitations & Review

- Validation data is used for both checkpoint selection and the reported evaluation; test-set performance is not reported.
- Fixed-length tokenization can truncate sequences longer than 256 tokens.
- The comparison does not include repeated runs, uncertainty estimates, per-entity error analysis, or inference-cost measurements.
- Dependency versions are not pinned, so changes to dataset loading or Trainer APIs may require compatibility adjustments.

The project demonstrates why tokenization and supervision must be aligned in NER. Masking later subwords preserves one supervised prediction per word, while the shared experiment setup supports comparison of the two pretrained models. Future work could add held-out test evaluation and entity-specific error analysis.

## Notes

The reported F1 scores are taken from the submitted assignment evaluation sheet. This portfolio repository reorganizes the original implementation for readability while preserving the experiment setup and evaluation logic.
