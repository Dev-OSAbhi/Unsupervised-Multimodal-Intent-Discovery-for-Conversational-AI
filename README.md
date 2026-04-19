# UMC — Unsupervised Multimodal Clustering

Implementation of **"Unsupervised Multimodal Clustering for Semantics Discovery in Multimodal Utterances"** (ACL 2024).

## Structure

```
umc/
├── configs/config.py          # Per-dataset hyper-parameters
├── data/dataloader.py         # Dataset loading (text/video/audio)
├── losses/losses.py           # Unsupervised & supervised contrastive losses
├── methods/umc/
│   ├── model.py               # UMCModel (BERT + V/A-Transformers + Fusion)
│   └── manager.py             # Pre-training, clustering, representation learning
├── utils/metrics.py           # NMI, ARI, ACC (Hungarian), FMI
├── run.py                     # Entry point
├── run_umc.sh                 # Shell script
└── requirements.txt
```

## Pipeline

**Step 1 – Multimodal Unsupervised Pre-training**
- Text: BERT → Linear → z_T
- Video: Swin features → Linear → Transformer → z_V
- Audio: WavLM features → Linear → Transformer → z_A
- Fusion: Cat(z_T, z_A, z_V) → GELU → z_TAV
- Two augmented views by zeroing one non-verbal modality: z_TA0, z_T0V
- Unsupervised contrastive loss over 3B augmented samples

**Step 2 – Clustering & High-Quality Sample Selection**
- K-Means++ on z_TAV (centroid inheritance from previous iteration)
- Density: ρ_i = K_near / Σ d_ij (reciprocal avg. dist to top-K neighbors)
- Auto K_near selection via intra-cluster cohesion score
- Curriculum threshold t = t0 + Δ × iter

**Step 3 – Multimodal Representation Learning**
- High-quality samples → supervised contrastive loss (head2)
- Low-quality samples → unsupervised contrastive loss (head3)
- Steps 2 & 3 iterate until t = 100%

## Setup

```bash
conda create -n umc python=3.8
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data

Download multimodal features (Swin Transformer for video, WavLM for audio) and place them as:
```
Datasets/
├── MIntRec/
│   ├── train.tsv / dev.tsv / test.tsv
│   ├── video_data/swin_feats.pkl
│   └── audio_data/wavlm_feats.pkl
├── MELD-DA/  ...
└── IEMOCAP-DA/  ...
```

## Run

```bash
bash run_umc.sh
# or
python run.py --dataset MIntRec --data_path Datasets --bert_path bert-base-uncased \
              --pretrain --train --save_model
```

## Results (from paper)

| Dataset     | NMI   | ARI   | ACC   | FMI   | Avg.  |
|-------------|-------|-------|-------|-------|-------|
| MIntRec     | 49.26 | 24.67 | 43.73 | 29.39 | 36.76 |
| MELD-DA     | 23.22 | 20.59 | 35.31 | 33.88 | 28.25 |
| IEMOCAP-DA  | 24.16 | 20.31 | 33.87 | 32.49 | 27.71 |
