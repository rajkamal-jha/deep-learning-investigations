# RollbackLROnPlateau

An experimental investigation of a rollback-based modification to
`ReduceLROnPlateau`.

The experiment asks whether restoring the best previously observed model
state before reducing the learning rate can produce a more favorable
optimization trajectory.

Two rollback variants are evaluated:

- **Rollback + Reset**: restore model parameters and reset optimizer state.
- **Rollback + Retain**: restore model parameters while retaining optimizer state.

Both are compared against the standard `ReduceLROnPlateau` scheduler across
10 independent Fashion-MNIST training runs.

## Main Results

| Configuration | Mean Best Val. Loss | Mean Epochs | Mean Training Time |
|---|---:|---:|---:|
| Rollback + Reset | 0.2753 | 45.7 | 136.2 s |
| Rollback + Retain | **0.2743** | 41.8 | 116.9 s |
| ReduceLROnPlateau | 0.2846 | 29.3 | 84.7 s |

Both rollback configurations achieved lower mean best validation loss than
the baseline in this experiment, at the cost of additional computation.

The results do not establish rollback as a universally superior scheduler.
A separate regression experiment showed similar performance between rollback
and `ReduceLROnPlateau`, suggesting that its effectiveness may depend on the
optimization dynamics of the task.

## Dataset

This experiment uses **Fashion-MNIST**, a dataset of 28×28 grayscale images
with 10 classes.

The dataset is **not included in this repository**. It can be obtained from
the original Fashion-MNIST repository:

- [Fashion-MNIST: Official Repository](https://github.com/zalandoresearch/fashion-mnist)
- [Fashion-MNIST Dataset Documentation](https://github.com/zalandoresearch/fashion-mnist#readme)

The original dataset contains 60,000 training examples and 10,000 test
examples. The experiment uses the training set with an 80/20 train-validation
split.

## Contents

```text
rollback_lr_scheduler/
│
├── figures/
├── results/
├── report/
│   └── rollback_lr_scheduler_report.pdf
│
├── README.md
├── rollback_lr_scheduler.ipynb
└── RollbackLROnPlateau.py
```

### Files

* **`rollback_lr_scheduler.ipynb`**
  Complete experimental notebook containing the setup, training runs,
  analysis, visualizations, and observations.

* **`RollbackLROnPlateau.py`**
  Custom rollback-based learning-rate scheduler.

* **`figures/`**
  Figures generated during the analysis.

* **`results/`**
  Raw and aggregated experimental results.

* **`report/`**
  Detailed research-style report containing the methodology, results,
  discussion, limitations, and conclusions.

## Experiment

The complete investigation covers:

* Rollback vs. standard `ReduceLROnPlateau`
* Optimizer state reset vs. retention
* Validation-loss behavior
* Learning-rate trajectories
* Rollback events
* Training-time trade-offs
* Run-to-run variation
* A separate regression counterexample

For the complete methodology and discussion, see the
[research report](report/rollback_lr_scheduler_report.pdf).
