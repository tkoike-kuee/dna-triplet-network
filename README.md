# Triplet Network-Based DNA Encoding for Enhanced Similarity Image Retrieval
## Overview

This repository contains the source code developed for our research paper:

**Title: Triplet Network-Based DNA Encoding for Enhanced Similarity Image Retrieval**  
Authors: Takefumi Koike, Hiromitsu Awano, Takashi Sato

Conference: DAC '24: Proceedings of the 61st ACM/IEEE Design Automation Conference

DOI: [https://doi.org/10.1145/3649329.3657320](https://doi.org/10.1145/3649329.3657320)

### Abstract
With the exponential growth of digital data, DNA is emerging as an attractive medium for storage and computing. Thus, design methods for encoding, storing, and searching digital data within DNA storage are of utmost importance. This paper introduces image classification as a measurable task for evaluating the performance of DNA encoders in similar image searches. Furthermore, we propose a novel triplet network-based DNA encoder to improve the accuracy and efficiency. The evaluation using the CIFAR-100 dataset demonstrates that the proposed encoder outperforms existing encoders in retrieving similar images, with an accuracy of 0.77, which is equivalent to 94% of the practical upper limit, and 16 times faster training time.

---

## Repository Structure

```
├── README.md
├── requirements.txt
├── primo                  # Core library inspired by https://github.com/uwmisl/primo-similarity-search
│   ├── __init__.py
│   ├── analysis
│   │   ├── __init__.py
│   │   └── decode.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── open_images.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── encoder.py
│   │   ├── encoder_trainer.py
│   │   ├── predictor.py
│   │   ├── simulator.py
│   │   └── triplet_network.py 
│   └── tools
│       ├── __init__.py
│       ├── barcoder.py
│       ├── calc_yields.py
│       ├── losses.py
│       ├── multiprogress.py
│       └── sequences.py
├── results.ipynb          # Jupyter notebook for analysis
├── run_simulations.py     # Script to run simulations
├── train_encoder.py       # Script to train the encoder model
└── train_predictor.py     # Script to train the predictor model
```

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/tkoike-kuee/dna-triplet-network.git
cd dna-triplet-network
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Data Preparation
[CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html)

### Training the Model
```bash
python train_encoder.py --train_data /path/to/train.h5 --test_data /path/to/test.h5 --target_seqs /path/to/target_seqs.h5 --query_seqs /path/to/query_seqs.h5 --encoder /path/to/encoder.h5 --loss 1
python run_simulations.py --target /path/to/target_seqs.h5 --query /path/to/query_seqs.h5 --output /path/to/simulations.h5
```

### Running a Jupyter Notebook (Optional)
```bash
jupyter notebook results.ipynb
```

---
