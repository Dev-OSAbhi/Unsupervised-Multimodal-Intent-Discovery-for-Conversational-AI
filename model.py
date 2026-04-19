import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MEncoder(nn.Module):
    """Transformer encoder for video or audio features."""
    def __init__(self, in_dim, dh, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, dh)
        enc_layer = nn.TransformerEncoderLayer(d_model=dh, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)

    def forward(self, x, mask=None):
        h = self.proj(x)
        h = self.transformer(h, src_key_padding_mask=mask)
        return h[:, -1, :]  # last sequence element


class FusionLayer(nn.Module):
    def __init__(self, dh, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(3 * dh, dh),
        )
        self.act = nn.GELU()

    def forward(self, zt, za, zv):
        cat = torch.cat([zt, za, zv], dim=-1)
        return self.act(self.net(cat))


class ConHead(nn.Module):
    def __init__(self, dh):
        super().__init__()
        self.fc = nn.Linear(dh, dh)

    def forward(self, x):
        return F.normalize(F.relu(self.fc(x)), dim=-1)


class UMCModel(nn.Module):
    def __init__(self, bert_path, vid_dim=1024, aud_dim=768, dh=256,
                 nhead=4, tf_layers=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.ft = nn.Linear(self.bert.config.hidden_size, dh)
        self.fv = MEncoder(vid_dim, dh, nhead, tf_layers, dropout)
        self.fa = MEncoder(aud_dim, dh, nhead, tf_layers, dropout)
        self.fusion = FusionLayer(dh, dropout)

        # three contrastive heads for pre-training, sup-cl, unsup-cl on low-qual
        self.head1 = ConHead(dh)  # pre-training
        self.head2 = ConHead(dh)  # supervised (high-quality)
        self.head3 = ConHead(dh)  # unsupervised (low-quality)

    def encode_text(self, ids, mask, seg):
        out = self.bert(input_ids=ids, attention_mask=mask, token_type_ids=seg)
        return self.ft(out.last_hidden_state[:, 0, :])

    def encode_video(self, v):
        return self.fv(v)

    def encode_audio(self, a):
        return self.fa(a)

    def get_repr(self, ids, mask, seg, v, a):
        zt = self.encode_text(ids, mask, seg)
        zv = self.encode_video(v)
        za = self.encode_audio(a)
        ztav = self.fusion(zt, za, zv)
        return zt, za, zv, ztav

    def get_aug_reprs(self, ids, mask, seg, v, a):
        """Returns (ztav, zta0, zt0v) - three augmentation views."""
        zt = self.encode_text(ids, mask, seg)
        zv = self.encode_video(v)
        za = self.encode_audio(a)

        zeros_v = torch.zeros_like(zv)
        zeros_a = torch.zeros_like(za)

        ztav = self.fusion(zt, za, zv)
        zta0 = self.fusion(zt, za, zeros_v)   # mask video
        zt0v = self.fusion(zt, zeros_a, zv)   # mask audio
        return ztav, zta0, zt0v

    def forward(self, ids, mask, seg, v, a):
        return self.get_repr(ids, mask, seg, v, a)
