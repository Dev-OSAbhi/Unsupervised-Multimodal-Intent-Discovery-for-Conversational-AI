# LearnMap — Automatic Intent Classification in Educational Conversations

LearnMap is a multimodal intent classification system built for educational dialogue. It processes text, audio, and video together to automatically discover what a speaker intends — whether they are explaining a concept, asking a question, seeking motivation, or giving advice. Trained on real conversational data from the web series *Aspirants*, it requires no labeled data and learns intent structure entirely on its own through multimodal contrastive learning and iterative clustering.

---

## Problem Statement

Classifying speaker intent in educational dialogues is a hard problem — and it gets harder when the only signal available is raw, unlabeled data. Text alone is often ambiguous; the same sentence can carry completely different intents depending on tone of voice or facial expression.

LearnMap tackles this by jointly learning from three modalities — text, audio, and video — and automatically grouping utterances into meaningful intent categories such as explaining, questioning, motivating, and advising, with no manual annotation required.

---

## Real-World Applications

| Domain | Application |
|---|---|
| Education | Automatically classify intent in recorded lectures and tutoring sessions |
| E-Learning Platforms | Understand learner and instructor intent from video course content |
| Virtual Assistants | Discover and expand intent coverage from real student interaction data |
| Academic Research | Map dialogue patterns and intent structures across educational corpora |
| Conversational Analytics | Analyze intent trends across large sets of educational video dialogues |

---

## System Architecture

```
Raw Conversation (text + audio + video)
          │
          ├──► BERT Encoder          → z_T  (text representation)
          ├──► WavLM + Transformer   → z_A  (audio representation)
          └──► Swin  + Transformer   → z_V  (video representation)
                          │
                    Fusion Layer
                    Cat(z_T, z_A, z_V) → GELU → z_TAV
                          │
          Stage 1: Multimodal Contrastive Pre-training
                   (augment by masking one modality at a time)
                          │
          Stage 2: K-Means++ Clustering + High-Quality Sample Selection
                   (curriculum density thresholding)
                          │
          Stage 3: Representation Learning
                   (supervised CL on confident samples,
                    unsupervised CL on ambiguous samples)
                          │
                   Iterate Stages 2 & 3
                          │
                  Discovered Intent Clusters
```

---

## Project Structure

```
LearnMap/
├── model.py                  # Core neural architecture
├── manager.py                # Training & evaluation manager
├── dataloader.py             # Multimodal data loading
├── losses.py                 # Contrastive loss functions
├── metrics.py                # Clustering evaluation metrics
├── config.py                 # Dataset-specific hyper-parameters
├── run.py                    # Main entry point
├── run_umc.sh                # Shell script for batch experiments
├── requirements.txt          # Python dependencies
├── .gitignore
└── Datasets/
    ├── MIntRec/              # 20 intent categories
    │   ├── train.tsv
    │   ├── dev.tsv
    │   ├── test.tsv
    │   ├── video_data/swin_feats.pkl
    │   └── audio_data/wavlm_feats.pkl
    ├── MELD-DA/              # 12 dialogue act categories
    │   └── ...
    └── IEMOCAP-DA/           # 12 dialogue act categories
        └── ...
```

### Key Components

| File | Role |
|---|---|
| `model.py` | UMCModel — BERT + Audio/Video Transformers + Fusion Layer |
| `manager.py` | Training orchestration — pre-training, clustering, representation learning |
| `dataloader.py` | Multimodal dataset loader (text/audio/video feature alignment) |
| `losses.py` | Unsupervised & supervised contrastive loss functions |
| `metrics.py` | Clustering evaluation — NMI, ARI, ACC (Hungarian), FMI |
| `config.py` | Hyper-parameter presets for each dataset |
| `run.py` | Entry point — parses arguments and launches the pipeline |
| `run_umc.sh` | Shell script to run all datasets |

---

## Setup & Installation

### 1. Create a Conda environment

```bash
conda create -n learnmap python=3.8
conda activate learnmap
```

### 2. Install PyTorch (CUDA 11.1)

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Pre-trained backbone models

LearnMap uses BERT-base-uncased (text), WavLM (audio), and Swin Transformer (video). BERT weights are downloaded automatically via HuggingFace. Pre-extracted audio and video features must be placed at:

```
Datasets/<DATASET>/video_data/swin_feats.pkl
Datasets/<DATASET>/audio_data/wavlm_feats.pkl
```

---

## Datasets

| Dataset | Intent / Act Categories | Domain |
|---|---|---|
| Aspirants | 20 | Educational dialogues (TVF web series) |
| MELD-DA | 12 | Dialogue acts (Friends TV corpus) |
| IEMOCAP-DA | 12 | Dialogue acts (dyadic session corpus) |

The primary dataset is sourced from the Hindi web series *Aspirants*, which follows students preparing for the UPSC civil services examination. Dialogues cover a wide range of educational intents including mentoring, explaining, questioning, motivating, and advising.

Each split file (`train.tsv`, `dev.tsv`, `test.tsv`) follows this format:

```
uid             text                                      label
ASP_E01_042     "yeh topic aise nahi samjhega..."         Explain
ASP_E02_017     "doubt hai mujhe is chapter mein."        Question
```

---

## Running the System

### Option A — Shell script (recommended)

```bash
# Run all datasets end-to-end
bash run_umc.sh

# Run a single dataset
bash run_umc.sh MIntRec

# Skip pre-training if weights are already saved
SKIP_PRETRAIN=1 bash run_umc.sh MIntRec
```

### Option B — Python directly

```bash
# Full pipeline: pre-train then train
python run.py \
    --dataset   MIntRec \
    --data_path Datasets \
    --bert_path bert-base-uncased \
    --pretrain \
    --train \
    --save_model

# Evaluation only (load saved weights)
python run.py \
    --dataset       MIntRec \
    --data_path     Datasets \
    --bert_path     bert-base-uncased \
    --pretrain_path outputs/MIntRec/pretrain.pt
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `MIntRec` | Dataset to use (`MIntRec`, `MELD-DA`, `IEMOCAP-DA`) |
| `--data_path` | `Datasets` | Root folder containing dataset directories |
| `--bert_path` | `bert-base-uncased` | HuggingFace model name or local path |
| `--output_path` | `outputs` | Where to save checkpoints and results |
| `--seed` | `0` | Random seed for reproducibility |
| `--pretrain` | flag | Run Stage 1 multimodal pre-training |
| `--train` | flag | Run Stages 2 & 3 clustering + representation learning |
| `--save_model` | flag | Save best model checkpoint |
| `--pretrain_path` | `None` | Load pre-trained weights from this path |
| `--batch_size` | from config | Override default batch size |

---

## Results

Evaluated using four standard unsupervised clustering metrics (higher is better):

| Metric | Description |
|---|---|
| NMI | Normalized Mutual Information — cluster purity vs. ground truth |
| ARI | Adjusted Rand Index — agreement between predicted and true groupings |
| ACC | Clustering accuracy with optimal Hungarian label assignment |
| FMI | Fowlkes-Mallows Index — geometric mean of precision and recall |

| Dataset | NMI | ARI | ACC | FMI | Avg |
|---|---|---|---|---|---|
| MIntRec | 49.26 | 24.67 | 43.73 | 29.39 | 36.76 |
| MELD-DA | 23.22 | 20.59 | 35.31 | 33.88 | 28.25 |
| IEMOCAP-DA | 24.16 | 20.31 | 33.87 | 32.49 | 27.71 |

All results are obtained without any labeled training data.

---

## How It Works

### Stage 1 — Multimodal Contrastive Pre-training

The model learns a shared embedding space across three views of each educational utterance:

- z_TAV — full multimodal fusion (text + audio + video)
- z_TA0 — text + audio only (video channel zeroed out)
- z_T0V — text + video only (audio channel zeroed out)

This forces the model to capture the core intent of an utterance regardless of which non-verbal channel is present — important for educational video where visual quality can vary.

### Stage 2 — Density-Based Sample Selection

After K-Means++ clustering, each sample is scored by local density:

```
rho_i = K_near / sum(distance to top-K neighbors)
```

Only the top-t% densest samples per cluster are treated as confident pseudo-labeled examples. The threshold follows a curriculum schedule:

```
t = t0 + delta x epoch
```

### Stage 3 — Dual Contrastive Representation Learning

- High-confidence samples use Supervised Contrastive Loss — pulling utterances with the same educational intent closer together.
- Low-confidence samples use Unsupervised Contrastive Loss — learning general structure without noisy pseudo-labels.

Stages 2 and 3 iterate until t reaches 100%.

## References

- Zhang et al., "Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances", ACL 2024
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", NAACL 2019
- Chen et al., "A Simple Framework for Contrastive Learning" (SimCLR), ICML 2020
- Liu et al., "Swin Transformer", ICCV 2021
- Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", IEEE JSTSP 2022

---
