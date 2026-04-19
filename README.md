# MultiSense AI — Multimodal Customer Intent Discovery System

> **Capstone Project** | AI & Machine Learning Engineering
> An unsupervised deep learning system that automatically discovers and groups customer intent categories from real-world conversational data — using speech, facial expressions, and spoken language together.

---

## 🎯 Problem Statement

In modern contact centers and virtual assistant platforms, understanding **what a customer truly wants** is the single most important challenge. Traditional approaches require human annotators to manually label thousands of hours of conversation recordings — an expensive, slow, and error-prone process.

**MultiSense AI** solves this by automatically discovering intent categories from raw multimodal conversations — **no labels required**. It simultaneously processes:

- 🗣️ **Text** — the words a customer speaks (transcript)
- 🎵 **Audio** — tone, pitch, and prosody of the voice (WavLM features)
- 🎥 **Video** — facial expressions and head gestures (Swin Transformer features)

By fusing all three signals, the system uncovers intent patterns that text alone would miss — e.g., a customer saying *"that's fine"* with an irritated tone is very different from saying it calmly.

---

## 🌍 Real-World Use Cases

| Domain | Application |
|---|---|
| **Contact Centers** | Auto-categorize inbound customer calls without pre-labeling |
| **Virtual Assistants** | Discover new user intents to expand the assistant's skill set |
| **Healthcare** | Understand patient intents in telehealth video consultations |
| **E-Learning** | Group student queries in recorded tutoring sessions |
| **Market Research** | Cluster consumer feedback from interview recordings |

---

## 🏗️ System Architecture

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
             ┌────────────┴────────────┐
             │   Three-Stage Pipeline  │
             └────────────┬────────────┘
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
                   ↺ Iterate Stages 2 & 3
                          │
                  Discovered Intent Clusters
```

### Key Components

| File | Role |
|---|---|
| `model.py` | `UMCModel` — BERT + Audio/Video Transformers + Fusion Layer |
| `manager.py` | Training orchestration — pre-training, clustering, representation learning |
| `dataloader.py` | Multimodal dataset loader (text/audio/video feature alignment) |
| `losses.py` | Unsupervised & supervised contrastive loss functions |
| `metrics.py` | Clustering evaluation — NMI, ARI, ACC (Hungarian), FMI |
| `config.py` | Hyper-parameter presets for each dataset |
| `run.py` | Entry point — parses arguments and launches the pipeline |
| `run_umc.sh` | Convenience shell script to run all datasets |

---

## 📂 Project Structure

```
MultiSense-AI/
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
    ├── MIntRec/              # 20 customer intent categories
    │   ├── train.tsv
    │   ├── dev.tsv
    │   ├── test.tsv
    │   ├── video_data/swin_feats.pkl
    │   └── audio_data/wavlm_feats.pkl
    ├── MELD-DA/              # 12 dialogue act categories (Friends TV)
    │   └── ...
    └── IEMOCAP-DA/           # 12 dialogue act categories (IEMOCAP)
        └── ...
```

---

## ⚙️ Setup & Installation

### 1. Create a Conda environment

```bash
conda create -n multisense python=3.8
conda activate multisense
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

### 4. Download pre-trained backbone models

The system uses **BERT-base-uncased** (text), **WavLM** (audio features), and **Swin Transformer** (video features). The audio and video features are pre-extracted and stored as `.pkl` files in the `Datasets/` folder.

```bash
# BERT weights are downloaded automatically by HuggingFace transformers
# Pre-extracted multimodal features must be placed at:
#   Datasets/<DATASET>/video_data/swin_feats.pkl
#   Datasets/<DATASET>/audio_data/wavlm_feats.pkl
```

---

## 🗂️ Datasets

| Dataset | # Intents / Acts | Domain | Modalities |
|---|---|---|---|
| **MIntRec** | 20 | Everyday conversational intents | Text · Audio · Video |
| **MELD-DA** | 12 | Dialogue acts (Friends TV show) | Text · Audio · Video |
| **IEMOCAP-DA** | 12 | Dialogue acts (dyadic sessions) | Text · Audio · Video |

### TSV Format

Each split file (`train.tsv`, `dev.tsv`, `test.tsv`) follows this schema:

```
uid    text                                    label
S05E16_329    "apparently, teens are drinking..."    Inform
S06E06_589    "we are the manager."                  Introduce
```

---

## 🚀 Running the System

### Option A — Shell script (recommended)

```bash
# Run all three datasets end-to-end
bash run_umc.sh

# Run a single dataset
bash run_umc.sh MIntRec

# Skip pre-training if weights are already saved
SKIP_PRETRAIN=1 bash run_umc.sh MIntRec
```

### Option B — Python directly

```bash
# Full pipeline: pre-train → cluster → evaluate
python run.py \
    --dataset   MIntRec \
    --data_path Datasets \
    --bert_path bert-base-uncased \
    --pretrain \
    --train \
    --save_model

# Evaluation only (using saved model)
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
| `--save_model` | flag | Save best model checkpoint to `output_path` |
| `--pretrain_path` | `None` | Load pre-trained weights from this path |
| `--batch_size` | from config | Override default batch size |

---

## 📊 Performance Results

Evaluated using **four standard unsupervised clustering metrics** (higher is better):

| Metric | Description |
|---|---|
| **NMI** | Normalized Mutual Information — measures cluster purity vs. true labels |
| **ARI** | Adjusted Rand Index — agreement between clustering and ground truth |
| **ACC** | Accuracy with optimal Hungarian label assignment |
| **FMI** | Fowlkes-Mallows Index — geometric mean of precision & recall |

### Results on Benchmark Datasets

| Dataset | NMI | ARI | ACC | FMI | **Avg** |
|---|---|---|---|---|---|
| MIntRec | 49.26 | 24.67 | 43.73 | 29.39 | **36.76** |
| MELD-DA | 23.22 | 20.59 | 35.31 | 33.88 | **28.25** |
| IEMOCAP-DA | 24.16 | 20.31 | 33.87 | 32.49 | **27.71** |

> These results are obtained **without any labeled training data**, making the system directly deployable on new, unseen conversation corpora.

---

## 🔬 Technical Deep-Dive

### Stage 1 — Multimodal Contrastive Pre-training

The model learns a shared embedding space by contrasting three views of each utterance:
- **z_TAV** — full multimodal fusion (text + audio + video)
- **z_TA0** — text + audio only (video masked with zeros)
- **z_T0V** — text + video only (audio masked with zeros)

This forces the model to capture the core semantic intent regardless of which non-verbal channel is available.

### Stage 2 — Density-Based High-Quality Sample Selection

After K-Means++ clustering, each sample is scored by its **local density**:

```
ρ_i = K_near / Σ(distance to top-K neighbors)
```

Only the top-t% densest samples per cluster (those closest to the cluster core) are treated as confident pseudo-labeled examples. The threshold `t` follows a **curriculum schedule**, starting low and gradually increasing:

```
t = t0 + Δ × epoch
```

### Stage 3 — Dual Contrastive Representation Learning

- **High-quality samples** → Supervised Contrastive Loss (pulls same-intent embeddings together)
- **Low-quality samples** → Unsupervised Contrastive Loss (learns general structure without noisy labels)

Stages 2 and 3 alternate until `t = 100%`.

---

## 💡 Why This Matters for Business

- **Zero labeling cost** — discovers intent categories from raw recordings automatically
- **Multimodal robustness** — captures what text alone misses (sarcasm, hesitation, frustration)
- **Scalable** — add new conversation data without retraining from scratch
- **Interpretable clusters** — each cluster = a discovered intent (e.g., *complaint*, *inquiry*, *approval*)
- **Domain-agnostic** — works on any domain with text + audio + video recordings

---

## 🛠️ Future Enhancements

- [ ] Real-time inference API (FastAPI / Flask endpoint)
- [ ] Interactive cluster visualization dashboard (UMAP + Plotly)
- [ ] Online learning — incrementally update clusters as new calls arrive
- [ ] Speaker diarization integration for multi-speaker conversations
- [ ] Export discovered intent taxonomy to downstream NLU systems (Rasa / Dialogflow)

---

## 📚 References

- Zhang et al., *"Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances"*, ACL 2024
- Devlin et al., *"BERT: Pre-training of Deep Bidirectional Transformers"*, NAACL 2019
- Chen et al., *"A Simple Framework for Contrastive Learning"* (SimCLR), ICML 2020
- Liu et al., *"Swin Transformer"*, ICCV 2021
- Chen et al., *"WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"*, IEEE JSTSP 2022

---

## 👥 Team

| Name | Role |
|---|---|
| *(your name)* | ML Engineering, Model Training, Evaluation |
| *(teammate)* | Data Pipeline, Feature Extraction |
| *(teammate)* | System Integration, Demo & Presentation |

---

*Capstone Project — submitted in partial fulfillment of the requirements for the degree in Artificial Intelligence & Machine Learning Engineering.*