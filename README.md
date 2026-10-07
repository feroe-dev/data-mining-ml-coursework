# Data Mining & Machine Learning Coursework

My coursework for **CS429 Data Mining** at the **University of Warwick** (Department of Computer Science, January–March 2021). The module covered classical machine learning (nearest neighbours, linear models, SVMs, PCA, model evaluation, clustering) and deep learning.

The repository contains my solutions to the two graded assignments.

| Project | Task | Methods |
| --- | --- | --- |
| [Assignment 1](assignment-1-image-classification/) | Binary image classification (28×28 images, labels ±1) | k-NN, preprocessing, PCA, kernel SVMs, stratified cross-validation, grid search |
| [Assignment 2](assignment-2-cell-counting/) | Count six cell types in 256×256 RGB microscopy patches (multi-output regression) | Hand-crafted colour features (RGB and HED space), OLS, MLP (scikit-learn and Keras), SVR, CNNs with transfer learning (MobileNet) |

## Assignment 1: Image classification

[`assignment_1.ipynb`](assignment-1-image-classification/assignment_1.ipynb) is the submitted notebook. Each `question_*.py` script holds the same code for one question.

1. **Data and metrics.** I explored the class balance (about 61% more positive than negative labels) and argued for AUC-PR over accuracy and AUC-ROC. I also derived the expected performance of a random classifier.
2. **k-nearest neighbours.** 5-fold stratified cross-validation, the effect of preprocessing (standardisation, normalisation) and the choice of k.
3. **Model comparison.** k-NN, perceptron, naive Bayes, logistic regression, and linear and kernelised SVMs, compared on accuracy, AUC-ROC and AUC-PR.
4. **PCA.** Scree plot analysis: 315 components explain 95% of the variance. I then trained classifiers on the reduced data.
5. **Final pipeline.** `StandardScaler → PCA → SVC` tuned with an 8-fold grid search over 360 configurations.

Final model: RBF-kernel SVM on 144 principal components (C = 8, γ = 0.001, balanced class weights).

| Metric (8-fold CV) | Mean | Std |
| --- | --- | --- |
| Accuracy | 0.846 | 0.013 |
| AUC-ROC | 0.923 | 0.013 |
| AUC-PR | 0.941 | 0.014 |

**Data:** `Xtrain`, `Ytrain` and `Xtest` are published in the course repository [foxtrotmike/CS909](https://github.com/foxtrotmike/CS909/tree/master/assignment1). Save them as `Xtrain.txt`, `Ytrain.txt` and `Xtest.txt` next to the scripts.

## Assignment 2: Cell counting in microscopy images

[`assignment_2.ipynb`](assignment-2-cell-counting/assignment_2.ipynb) is the submitted notebook, and the `question_*.py` scripts hold the per-question code. Folds 1 and 2 are used for training and validation, and fold 3 for testing.

1. **Data analysis.** Per-fold statistics, cell-count distributions, and correlations between image features and cell counts.
2. **Feature-based regression.** I extracted mean, variance and entropy from the haematoxylin (H) channel and the RGB channels, then compared OLS, MLPs and support vector regression.
3. **Deep learning.** First a CNN with my own preprocessing, then a MobileNet-based CNN that I evaluated with cross-validation over all three folds and per cell type.

Results on the test fold. A higher Pearson correlation is better and a lower RMSE is better.

| Model | RMSE | Pearson r | Spearman ρ | R² |
| --- | --- | --- | --- | --- |
| OLS (H-channel average) | 9.94 | 0.581 | 0.575 | 0.312 |
| MLP (H-channel average) | 10.42 | 0.527 | 0.492 | 0.259 |
| SVR (H-channel average) | 10.15 | 0.573 | 0.608 | 0.300 |
| CNN, own preprocessing | 8.93 | 0.612 | 0.537 | 0.370 |
| **CNN, MobileNet** | **5.59** | **0.871** | **0.827** | **0.753** |
| CNN, MobileNet (mean over 3 folds) | 6.10 | 0.880 | 0.828 | 0.728 |

The notebook's consolidation table also covers every feature, every model and a per-cell-type breakdown.

**Data:** [`cell-data.npz`](https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/cell-data.npz) (about 440 MB) comes from the module page and is not included here. The scripts for Question 2 also reload intermediate results from a local `dill` session file, `PickleAssignment2.db`, which is too large to include.

## Setup

The code was written in 2021 with Python 3.7, scikit-learn and TensorFlow 2.4.1.

```bash
pip install -r requirements.txt
```

## Notes

The assignment specifications, lecture material and datasets belong to the University of Warwick and the module staff, so they are not included in this repository. The code is shown as it was submitted.
