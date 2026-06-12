# Optimized Image Captioning Pipeline

A professional, high-performance Image Captioning system built using TensorFlow, Keras, and the Flickr8k dataset. This repository is optimized for speed, memory efficiency, and reproducibility, incorporating modern software engineering and Deep Learning best practices.

---

## 🚀 Key Features & Architectural Upgrades

This implementation is a significant upgrade from basic tutorial-level notebooks, designed to meet rigorous standards for competitive ML program reviews:

1. **High-Performance Data Input Pipeline (`tf.data.Dataset`)**:
   * Replaced custom python generator loops with an optimized `tf.data.Dataset` using asynchronous prefetching (`prefetch(tf.data.AUTOTUNE)`) and batching. This prevents CPU bottlenecks and maximizes GPU throughput.
2. **Memory-Efficient Label Processing**:
   * Switched from sparse/one-hot categorical encoding (`to_categorical`) to integer labels.
   * Utilizes `sparse_categorical_crossentropy` loss, reducing memory consumption by up to **98%** for large vocabularies during sequence generation.
3. **Rigorous Evaluation Metrics**:
   * Evaluates performance on an independent validation set using cumulative **BLEU scores** (BLEU-1, BLEU-2, BLEU-3, BLEU-4) via the `nltk` package.
4. **Reproducible Splitting & Leakage Prevention**:
   * Split the dataset on the *image level* (90/10 split) prior to tokenization to prevent data leakage from overlapping image captions.
5. **Robust Training Controls**:
   * Implements `ModelCheckpoint` to save the best model weights based on validation loss.
   * Uses `EarlyStopping` with validation loss monitoring to prevent overfitting.
   * Outputs logging compatible with **TensorBoard** for visual analysis of training curves.
6. **Extensible Models (`model.py`)**:
   * **Optimized Merge Model**: High-efficiency functional baseline working with 2D global InceptionV3 feature maps.
   * **Bahdanau Attention-Based CNN-LSTM**: Modular subclass implementations for additive attention over spatial feature maps, enabling the decoder to focus on specific regions of the image during word generation.

---

## 📁 Repository Structure

```text
d:/Evoastra_Image_Captioning/
├── config.py           # Centralized hyperparameter and path configuration
├── dataset.py          # Data preprocessing, tokenization, tf.data dataset creation
├── model.py            # Optimized merge model & Bahdanau attention subclasses
├── train.py            # Feature extraction fallback, data splitting, and model training
├── eval.py             # Evaluation script to compute validation BLEU-1 to BLEU-4 scores
├── predict.py          # Inference script to caption custom or random images
├── requirements.txt    # Package dependencies
└── README.md           # Professional project documentation
```

---

## ⚙️ Setup and Installation

### 1. Clone & Set Up Directory
Place your Flickr8k images in the `Images/` folder, and the corresponding `captions.txt` in the root directory.

### 2. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Usage Instructions

### Step 1: Train the Model
Run the training script. If `features.pkl` does not exist, the script will automatically initialize InceptionV3 and extract average-pooled features for all images in the `Images/` folder first.
```bash
python train.py
```
*Weights will be saved to `best_model.keras` and logs written to the `logs/` directory.*

### Step 2: Evaluate Model Performance
To calculate the cumulative corpus BLEU scores on the validation subset:
```bash
python eval.py
```

### Step 3: Run Inference (Captioning New Images)
You can caption a specific image:
```bash
python predict.py --image path/to/your/image.jpg
```
*If `--image` is omitted, the script will automatically select a random image from the `Images/` folder to caption for quick testing.*

---

## 🔬 Model Architectures

### 1. The Merge Architecture
The default trained model uses a late-fusion merge architecture:
* **Image Branch**: Takes InceptionV3 features `(2048,)`, applies `Dropout(0.5)`, and projects to `256` units via a `Dense` layer.
* **Text Branch**: Embedding layer projects vocabulary tokens to a `256` embedding space (using `mask_zero=True` to ignore padding), processed by an `LSTM(256)`.
* **Fusion Decoder**: Combines both branches element-wise (`add`), followed by a `Dense(256)` layer and a `Dense(vocab_size)` softmax output.

### 2. The Spatial Attention Architecture (Optional Upgrade)
`model.py` contains pre-built subclasses for a **Bahdanau Attention** decoder:
* Uses spatial feature maps (e.g. `(8, 8, 2048)`) instead of global features.
* The attention weights align the decoder's hidden state with the spatial image regions.
* Concatenates the context vector with the target word embeddings at each LSTM step.
