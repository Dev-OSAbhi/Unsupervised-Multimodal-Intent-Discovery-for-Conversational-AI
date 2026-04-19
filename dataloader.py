import os, pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class MMDataset(Dataset):
    def __init__(self, data, vid_feats, aud_feats, vl, al, mode='train'):
        self.text_ids = data['text_ids']
        self.text_mask = data['text_mask']
        self.text_seg = data['text_seg']
        self.uids = data['uids']
        self.vid_feats = vid_feats
        self.aud_feats = aud_feats
        self.vl = vl
        self.al = al
        self.labels = data['labels']
        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        vf = self.vid_feats.get(uid, np.zeros((1, 1024), dtype=np.float32))
        af = self.aud_feats.get(uid, np.zeros((1, 768), dtype=np.float32))
        vf = pad_or_trunc(vf, self.vl)
        af = pad_or_trunc(af, self.al)
        
        return {
            'text_ids': torch.tensor(self.text_ids[idx], dtype=torch.long),
            'text_mask': torch.tensor(self.text_mask[idx], dtype=torch.long),
            'text_seg': torch.tensor(self.text_seg[idx], dtype=torch.long),
            'video': torch.tensor(vf, dtype=torch.float),
            'audio': torch.tensor(af, dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'idx': idx,
        }


def pad_or_trunc(feat, max_len):
    L = feat.shape[0]
    if L >= max_len:
        return feat[:max_len]
    pad = np.zeros((max_len - L, feat.shape[1]), dtype=np.float32)
    return np.concatenate([feat, pad], axis=0)


def load_mm_data(data_path, dataset, bert_path, seq_lens, seed=0):
    base = os.path.join(data_path, dataset)
    tok = BertTokenizer.from_pretrained(bert_path)

    with open(os.path.join(base, 'video_data', 'swin_feats.pkl'), 'rb') as f:
        vid_feats = pickle.load(f)
    with open(os.path.join(base, 'audio_data', 'wavlm_feats.pkl'), 'rb') as f:
        aud_feats = pickle.load(f)

    tl, vl, al = seq_lens

    splits = {}
    label_map = {}
    for split in ['train', 'dev', 'test']:
        fpath = os.path.join(base, f'{split}.tsv')
        if not os.path.exists(fpath):
            continue
        ids, masks, segs, uids, labs = [], [], [], [], []
        with open(fpath, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            parts = line.strip().split('\t')
            uid, text, label_str = parts[0], parts[1], parts[-1]
            try:
                label = int(label_str)
            except ValueError:
                if label_str not in label_map:
                    label_map[label_str] = len(label_map)
                label = label_map[label_str]
            enc = tok(text, max_length=tl, padding='max_length',
                      truncation=True, return_tensors='np')
            ids.append(enc['input_ids'][0])
            masks.append(enc['attention_mask'][0])
            segs.append(enc['token_type_ids'][0])
            uids.append(uid)
            labs.append(label)
        splits[split] = {
            'text_ids': np.array(ids),
            'text_mask': np.array(masks),
            'text_seg': np.array(segs),
            'uids': uids,
            'labels': np.array(labs),
        }

    # merge train+dev for unsupervised clustering
    if 'dev' in splits:
        merged = {}
        for k in splits['train']:
            if k == 'uids':
                merged[k] = splits['train'][k] + splits['dev'][k]
            else:
                merged[k] = np.concatenate([splits['train'][k], splits['dev'][k]], axis=0)
        splits['train'] = merged

    return splits, vid_feats, aud_feats


def get_loaders(data_path, dataset, bert_path, seq_lens, batch_size, seed=0):
    splits, vid_feats, aud_feats = load_mm_data(data_path, dataset, bert_path, seq_lens, seed)
    tl, vl, al = seq_lens
    loaders = {}
    for sp, data in splits.items():
        ds = MMDataset(data, vid_feats, aud_feats, vl, al, mode=sp)
        shuffle = (sp == 'train')
        loaders[sp] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=0, pin_memory=True, drop_last=False)
    return loaders
