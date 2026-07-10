# Model Error Analysis: Confusion Matrix and Classification Metrics

This project implements multiclass classification error analysis from first principles with NumPy: building a confusion matrix from one-hot encoded labels and predictions, then deriving per-class sensitivity (recall), precision, specificity, and F1 score directly from that matrix using fully vectorized array operations, with no explicit per-class loops and no external ML metrics library.

## Overview

Accuracy alone hides how a model actually fails. A classifier can reach high overall accuracy while being systematically wrong on a minority class, or while consistently trading false positives for false negatives in a way that matters a lot for the task at hand (e.g. missed detections vs. false alarms). The confusion matrix is the object that preserves this information: it tells you, for every true class, exactly which class the model predicted instead. Sensitivity, precision, specificity, and F1 are just different projections of that matrix, each answering a different question:

- **Sensitivity (recall)** — of the actual members of a class, how many did the model find?
- **Precision** — of the points the model labeled as a class, how many were correct?
- **Specificity** — of the actual non-members of a class, how many did the model correctly reject?
- **F1** — the harmonic mean of precision and sensitivity, useful when both false positives and false negatives are costly and classes are imbalanced.

Computing all of these per class as vectorized operations over the confusion matrix (row sums, column sums, and the diagonal) rather than iterating class-by-class is both more efficient and less error-prone: each metric reduces to one array expression that produces a `(classes,)` vector in a single pass.

## Contents

| File | Description |
| --- | --- |
| `0-create_confusion.py` | `create_confusion_matrix(labels, logits)` — builds a `(classes, classes)` confusion matrix from one-hot `labels` and one-hot `logits` by converting both to class indices and tallying occurrences. |
| `1-sensitivity.py` | `sensitivity(confusion)` — computes recall for every class as `diagonal / row_sum`, returning a `(classes,)` vector in one vectorized division. |
| `2-precision.py` | `precision(confusion)` — computes precision for every class as `diagonal / column_sum`. |
| `3-specificity.py` | `specificity(confusion)` — computes specificity for every class by deriving true negatives from the matrix total, row sums, column sums, and the diagonal. |
| `4-f1_score.py` | `f1_score(confusion)` — computes the per-class F1 score as the harmonic mean of the `1-sensitivity` and `2-precision` outputs. |
| `5-error_handling` | Written answers diagnosing four bias/variance regimes (high bias & high variance, high bias & low variance, low bias & high variance, low bias & low variance) and selecting the appropriate remediation for each — e.g. training a bigger/longer model or searching a better architecture to address avoidable bias, versus gathering more data or adding regularization to address variance. |
| `6-compare_and_contrast` | A written answer identifying the dominant issue (avoidable bias vs. variance) in a model by comparing its training and validation confusion matrices against a stated human-level error rate (~14%) used as a Bayes error proxy. |

## How It Works

### Confusion matrix construction

`labels` and `logits` arrive as one-hot arrays of shape `(m, classes)`. Both are collapsed to class indices with `np.argmax(..., axis=1)`, giving the true class and the predicted class for each of the `m` data points. The confusion matrix is a `(classes, classes)` array where row `i`, column `j` counts how many points with true class `i` were predicted as class `j`; it is filled by incrementing `confusion[true_class][predicted_class]` for every sample, so the diagonal holds the correct predictions and every off-diagonal cell is a specific type of misclassification.

### Vectorized per-class metrics

Given the confusion matrix `C`, every metric is read off it directly, for all classes at once:

- `row_sum = C.sum(axis=1)` — total actual instances of each class (true positives + false negatives).
- `col_sum = C.sum(axis=0)` — total predicted instances of each class (true positives + false positives).
- `diag = np.diag(C)` — true positives per class.
- `total = C.sum()` — total number of samples.

From these:

- **Sensitivity** = `diag / row_sum` → `TP / (TP + FN)`
- **Precision** = `diag / col_sum` → `TP / (TP + FP)`
- **Specificity** = `(total - row_sum - col_sum + diag) / (total - row_sum)` → `TN / (TN + FP)`, where true negatives are recovered as everything outside the class's row and column, with the diagonal cell added back once since it was subtracted twice.
- **F1 score** = `2 * precision * sensitivity / (precision + sensitivity)`, computed by importing and reusing the `1-sensitivity` and `2-precision` modules rather than recomputing the ratios inline.

Each formula operates on the whole `(classes,)` vectors simultaneously, so the result for all classes is produced by a single arithmetic expression.

## Requirements

- Python 3.12
- NumPy 2.4.5

## Usage

```python
#!/usr/bin/env python3
import numpy as np

create_confusion_matrix = __import__('0-create_confusion').create_confusion_matrix
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision
specificity = __import__('3-specificity').specificity
f1_score = __import__('4-f1_score').f1_score

lib = np.load('../../data/labels_logits.npz')
labels = lib['labels']
logits = lib['logits']

confusion = create_confusion_matrix(labels, logits)
print(confusion)          # (classes, classes) matrix of counts
print(sensitivity(confusion))  # (classes,) recall per class
print(precision(confusion))    # (classes,) precision per class
print(specificity(confusion))  # (classes,) specificity per class
print(f1_score(confusion))     # (classes,) F1 per class
```

Each metric function expects a plain `(classes, classes)` confusion matrix (for example loaded back from a saved `confusion.npz`) and returns a `(classes,)` NumPy array with one value per class, so results can be inspected directly or fed into further aggregation (macro/weighted averages, etc.).

## Design Notes

- All four downstream metrics are pure functions of the confusion matrix — none of them touch the raw labels or logits again — which keeps the confusion matrix as the single source of truth for every derived statistic, mirroring how tools like scikit-learn's `classification_report` are built internally.
- Every metric is computed as one vectorized NumPy expression over row sums, column sums, and the diagonal instead of looping over classes, so the cost stays proportional to the size of the confusion matrix rather than the number of classes times the number of samples.
- `f1_score` deliberately imports and reuses `sensitivity` and `precision` instead of re-deriving true positives/false positives/false negatives, avoiding duplicated logic and keeping a single implementation of each underlying ratio.
- Specificity is derived algebraically from quantities already available on the matrix (total, row sum, column sum, diagonal) rather than by explicitly constructing a one-vs-rest binary confusion matrix per class, which avoids reshaping or looping over classes entirely.

## Author

Fjolla Qerimi
