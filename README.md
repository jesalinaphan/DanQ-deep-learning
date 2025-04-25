# Overview 
- DanQ is a deep learning model that predicts the function of non-coding DNA directly from raw sequence. It uses a hybrid architecture that combines convolutional neural networks (to detect local motifs) with bi-directional recurrent layers (to capture dependencies between motifs). This design enables the model to learn both the building blocks and the regulatory grammar of gene expression.
- The problem is formulated as a multi-label classification task: given a DNA sequence, the model outputs a binary vector indicating the presence or absence of 919 different epigenetic marks.

# DanQ architecture:
- Input Layer: A 1000 base-pair DNA sequence is one-hot encoded into a 1000 × 4 binary matrix (one row per base pair, with four columns for A, C, G, T)
- Convolutional Layer: Applies 320 convolutional filters across the sequence to scan for local motifs. Each filter acts as a motif detector, learning biologically relevant patterns.
- Max pooling layer: Reduces the dimensionality of the convolutional output while preserving the strongest activations.
- Bidirectional LSTM layer: Processes the pooled feature map in both directions to learn dependencies between motifs
- Fully Connected Dense Layer: The outputs of the BLSTM are flattened and passed into a dense layer with rectified linear units (ReLU), enabling the model to learn higher-order combinations of features.
- Output Layer: A final sigmoid layer produces a 919-dimensional output vector, where each element represents the probability of a specific epigenetic mark being present in the input sequence. These labels include transcription factor binding, DNase sensitivity, and histone modifications
		
The max pooling and BiLSTM layers also use dropouts of 0.2 and 0.5,
respectively, to regularize the model

# Usage 
---

### **Step 1: Download the mm10 genome and ENCODE peak datasets**

```bash
./data/data_download.sh
```

---

### **Step 2: Generate non-overlapping 200bp bins**

```bash
./data/gen_200bp.sh
```

> ⚠️ Requires `bedtools` and `samtools` to be installed.

---

### **Step 3: Generate `preprocessed.csv` by running**

```bash
python3 ./preprocessing/preprocessing.py
```

---

Once you have access to ./data/processed.csv, running 

```bash
python3 main.py
```
will train and evaluate the DanQ model 
