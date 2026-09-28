from seqeval.metrics import f1_score, precision_score, recall_score


def compute_metrics(eval_prediction, label_list):
    """Compute token-classification metrics while ignoring masked labels."""
    predictions, labels = eval_prediction
    predictions = predictions.argmax(axis=-1)

    true_predictions = [
        [
            label_list[prediction]
            for prediction, label in zip(pred_row, label_row)
            if label != -100
        ]
        for pred_row, label_row in zip(predictions, labels)
    ]
    true_labels = [
        [
            label_list[label]
            for _prediction, label in zip(pred_row, label_row)
            if label != -100
        ]
        for pred_row, label_row in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
