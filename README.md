# Machine Learning: CNN and Text Classification

Two machine-learning projects in Python from my Computer Science degree at Wilfrid Laurier University (2024), built in a 3-person team with Harris Aman and Abdul Hafeez Mohammad.

## 1. CNN features, PCA and k-NN on MNIST (`cnn-pca-knn/`)

- Trains a compact **Keras CNN** (two convolution and two max-pooling layers, 9,098 parameters) on the 60,000 MNIST training images, reaching **98.64%** test accuracy.
- Extracts the CNN's 784-dimensional features from its Flatten layer, visualizes them in 2D with **PCA**, and classifies 10-component PCA features (fit on training data only) with **k-NN** (k = 5) at **95.48%** test accuracy.

MNIST downloads automatically through `keras.datasets`.

```bash
cd cnn-pca-knn
python cnn_pca_knn.py
```

## 2. Decision-tree fake-news classifier (`decision-tree-fake-news/`)

- Classifies 3,266 news headlines (1,968 real, 1,298 fake) as real or fake using bag-of-words features (**CountVectorizer**) and **scikit-learn** decision trees.
- Tunes `max_depth` over 28 values on a validation split; the best tree reached **76.33%** test accuracy. It also renders the tree with Graphviz.

The headline files are not included. Put the course's `real.txt` and `fake.txt` (one headline per line) in this folder, and install the [Graphviz](https://graphviz.org/download/) executable:

```bash
cd decision-tree-fake-news
python decision_tree.py
```

## Setup

```bash
pip install -r requirements.txt
```
