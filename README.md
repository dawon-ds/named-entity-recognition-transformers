# Named Entity Recognition with RoBERTa & ALBERT

[Portfolio Page](https://app.notion.com/p/3e968564df5a81ffa20edbcd1d169c0e)

Fine-tuning **RoBERTa-base** and **ALBERT-base-v2** for named entity recognition (NER) on **CoNLL-2003** and comparing their validation F1 scores.

## Project Overview

- **Task:** Named Entity Recognition / Token Classification
- **Dataset:** CoNLL-2003
- **Models:** RoBERTa-base, ALBERT-base-v2
- **Tech Stack:** Python, PyTorch, Hugging Face Transformers, Datasets, seqeval

## Implementation

- Loads the CoNLL-2003 dataset and tokenizes its pre-split words.
- Aligns word-level NER tags with subword tokens using `word_ids()`.
- Assigns each word's label to its first subword; masks later subwords, special tokens, and padding with `-100`.
- Fine-tunes both models using Hugging Face Trainer and the same hyperparameters.
- Computes entity-level Precision, Recall, and F1 using `seqeval`.
- Selects checkpoints using validation F1 and saves training plots.

## Experiment Settings

| Model | Epochs | Batch Size | Learning Rate | Max Length |
| --- | ---: | ---: | ---: | ---: |
| RoBERTa-base | 3 | 16 | 2e-5 | 256 |
| ALBERT-base-v2 | 3 | 16 | 2e-5 | 256 |

**Weight Decay:** 0.01 · **Seed:** 42

## Results

| Model | Validation F1 |
| --- | ---: |
| RoBERTa-base | **0.9588** |
| ALBERT-base-v2 | 0.9394 |

RoBERTa-base achieved a validation F1 **1.94 percentage points** higher than ALBERT-base-v2 under the compared settings.

## Repository Structure

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

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run either model:

```bash
python main.py --config config/roberta.yaml
python main.py --config config/albert.yaml
```

The program loads CoNLL-2003 and pretrained model/tokenizer files through Hugging Face. Checkpoints and training figures are written to `outputs/roberta/` or `outputs/albert/`, including `loss_plot.png` and `f1_plot.png`.

**Data and models:** CoNLL-2003 and the pretrained checkpoints are third-party resources with their respective usage and license terms.
