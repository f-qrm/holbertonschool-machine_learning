# Machine Learning Portfolio

This repository is a from-first-principles machine learning portfolio: mathematical foundations (linear algebra, calculus, probability) and supervised learning systems (classifiers, CNNs, object detection, neural style transfer, transfer learning) implemented mostly with NumPy alone, then connected to their TensorFlow/Keras equivalents where a production framework is the right tool. It was built while completing Holberton School's Machine Learning specialization and is maintained here as a working record of that foundation.

The organizing idea across the whole repository is the same one used in each individual project: before relying on a library call, implement the operation by hand — matrix multiplication, gradient descent, convolution, non-max suppression, dropout — so that the shapes, formulas, and edge cases a framework normally hides become explicit. Where a framework equivalent exists (Keras optimizers, layers, callbacks), it is used and compared directly against the manual version.

## Repository Structure

| Directory | Contents |
| --- | --- |
| [`math/`](math/README.md) | Linear algebra, advanced linear algebra (determinants/inverses), calculus, probability distributions, multivariate probability, plotting, and convolution/pooling — the mathematical building blocks used throughout the rest of the repository. |
| [`supervised_learning/`](supervised_learning/README.md) | Classifiers built from a single neuron up to a deep neural network, decision trees/random forests/isolation forests, error analysis, optimization and regularization techniques, the Keras API, CNN internals, ResNet-50, YOLOv3 object detection, neural style transfer, and transfer learning with MobileNetV2. |
| `data/` | Local datasets (MNIST, CIFAR-10-derived, binary classification splits) and a saved model, used by the driver scripts across `supervised_learning/`. Not tracked as project deliverables. |

Each subdirectory has its own README describing what was implemented, how it works, and the reasoning behind specific design choices — start there for the technical detail on any given topic.

## Tech Stack

- **Python** 3.12
- **NumPy** 2.4.5 — used as the near-exclusive dependency for every "from scratch" implementation (linear algebra, probability, classifiers, CNN forward/backward passes, decision trees).
- **TensorFlow** 2.21.0 / **Keras** 3.14.1 — used once a concept has been implemented manually, to show the idiomatic framework equivalent (optimizers, layers, callbacks, pretrained architectures).
- **Matplotlib** 3.10.9 — visualization and exploratory data analysis.
- **OpenCV** (`cv2`), **scikit-learn**, **Pillow**, **h5py** — used in specific projects (object detection, decision trees, neural style transfer, model serialization respectively) where noted in their own READMEs.

## Getting Started

```bash
git clone git@github.com:f-qrm/holbertonschool-machine_learning.git
cd holbertonschool-machine_learning
python3 -m venv venv
source venv/bin/activate
pip install numpy tensorflow matplotlib pillow h5py opencv-python scikit-learn
```

Every deliverable is a standalone, importable Python module (no package-level `__init__.py`); each is exercised by a matching `N-main.py` driver script in the same directory, which is the fastest way to see any given function or class in action — see the individual project READMEs for concrete usage examples.

## Author

**Fjolla Qerimi**
GitHub: [f-qrm](https://github.com/f-qrm)
