# Named Entity Recognition with RoBERTa & ALBERT

A token-classification project comparing **RoBERTa-base** and **ALBERT-base-v2** on the **CoNLL-2003 Named Entity Recognition (NER)** dataset.

## Overview

The project fine-tunes pretrained Transformer models under the same training setup and compares their validation F1 scores.

| Model | Validation F1 |
| --- | ---: |
| RoBERTa-base | **0.9588** |
| ALBERT-base-v2 | **0.9394** |

Both experiments used 3 epochs, batch size 16, learning rate 2e-5, and weight decay 0.01.

## What I Implemented

- Loaded the CoNLL-2003 NER dataset with Hugging Face Datasets.
- Tokenized pre-split words with a pretrained tokenizer.
- Aligned word-level NER labels to subword tokens.
- Ignored special tokens and repeated subword pieces with label `-100`.
- Fine-tuned Transformer token-classification models with Hugging Face Trainer.
- Evaluated precision, recall, and F1 using `seqeval`.
- Compared RoBERTa-base and ALBERT-base-v2 under the same training configuration.

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

```bash
python main.py --config config/roberta.yaml
python main.py --config config/albert.yaml
```

## Notes

The reported F1 scores are taken from the submitted assignment evaluation sheet. This portfolio repository reorganizes the original implementation for readability while preserving the experiment setup and evaluation logic.
