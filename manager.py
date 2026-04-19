import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.cluster import KMeans
from transformers import get_linear_schedule_with_warmup

from model import UMCModel
from losses import UnsupConLoss, SupConLoss
from metrics import calc_metrics


def get_feats(model, loader, device):
    model.eval()
    all_z, all_y, all_idx = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch['text_ids'].to(device)
            mask = batch['text_mask'].to(device)
            seg = batch['text_seg'].to(device)
            v = batch['video'].to(device)
            a = batch['audio'].to(device)
            _, _, _, z = model(ids, mask, seg, v, a)
            all_z.append(z.cpu().numpy())
            all_y.append(batch['label'].numpy())
            all_idx.append(batch['idx'].numpy())
    return (np.concatenate(all_z), np.concatenate(all_y),
            np.concatenate(all_idx))


def kmeans_pp(feats, k, centroids=None, seed=0):
    km = KMeans(n_clusters=k, init='k-means++' if centroids is None else centroids,
                n_init=1 if centroids is not None else 10, random_state=seed)
    km.fit(feats)
    return km.labels_, km.cluster_centers_


def calc_density(feats, k_near):
    """Compute density as Knear / sum(dist to top-K neighbors)."""
    n = len(feats)
    dists = np.linalg.norm(feats[:, None] - feats[None, :], axis=-1)  # (n,n)
    np.fill_diagonal(dists, np.inf)
    rho = np.zeros(n)
    for i in range(n):
        top_k = np.sort(dists[i])[:k_near]
        rho[i] = k_near / (top_k.sum() + 1e-8)
    return rho


def cohesion_score(feats, idxs):
    m = len(idxs)
    if m <= 1:
        return 0.0
    sub = feats[idxs]
    total = 0.0
    for i in range(m):
        diffs = np.linalg.norm(sub[i] - sub, axis=-1)
        total += diffs.sum() / (m - 1)
    return total / m


def select_high_quality(feats, labels, k, t, L=0.1, delta_k=0.02, u=10):
    """
    For each cluster, auto-select K_near via cohesion, then pick top-t% by density.
    Returns dict: cluster_id -> selected_indices (global)
    """
    selected = {}
    for c in range(k):
        cidx = np.where(labels == c)[0]
        n = len(cidx)
        if n == 0:
            selected[c] = np.array([], dtype=int)
            continue
        m = max(1, int(n * t))

        # auto select best K_near
        candidates = [max(1, int(n * (L + delta_k * q))) for q in range(u)]
        candidates = list(set(candidates))
        best_coh, best_k = -1, candidates[0]
        for kk in candidates:
            kk = min(kk, n - 1) if n > 1 else 1
            rho = calc_density(feats[cidx], kk)
            sorted_idx = np.argsort(-rho)
            chosen = cidx[sorted_idx[:m]]
            coh = cohesion_score(feats, chosen)
            if coh > best_coh:
                best_coh = coh
                best_k = kk

        rho = calc_density(feats[cidx], best_k)
        sorted_idx = np.argsort(-rho)
        selected[c] = cidx[sorted_idx[:m]]
    return selected


class UMCManager:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = UMCModel(
            bert_path=args.bert_path,
            dh=args.dh,
        ).to(device)
        self.k = args.num_labels
        self.centroids = None

    # ---------------------------------------------------------------
    # Step 1: Pre-training with multimodal unsupervised contrastive loss
    # ---------------------------------------------------------------
    def pretrain(self, loader):
        args = self.args
        ucl = UnsupConLoss(tau=args.tau1).to(self.device)

        bert_params = list(self.model.bert.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if 'bert' not in n]
        opt = AdamW([
            {'params': bert_params, 'lr': args.lr_pre},
            {'params': other_params, 'lr': args.lr_pre * 10},
        ], weight_decay=1e-2)
        total_steps = args.epochs_pre * len(loader)
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)

        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        for ep in range(args.epochs_pre):
            total_loss = 0
            opt.zero_grad()
            for step, batch in enumerate(loader):
                ids = batch['text_ids'].to(self.device)
                mask = batch['text_mask'].to(self.device)
                seg = batch['text_seg'].to(self.device)
                v = batch['video'].to(self.device)
                a = batch['audio'].to(self.device)

                with torch.cuda.amp.autocast():
                    ztav, zta0, zt0v = self.model.get_aug_reprs(ids, mask, seg, v, a)
                    z_all = torch.cat([ztav, zta0, zt0v], dim=0)
                    h = self.model.head1(z_all)
                    loss = ucl(h)
                    loss = loss / args.grad_acc_steps

                scaler.scale(loss).backward()
                
                if (step + 1) % args.grad_acc_steps == 0 or (step + 1) == len(loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()

                total_loss += loss.item() * args.grad_acc_steps

            if (ep + 1) % 5 == 0:
                print(f'[Pretrain] Epoch {ep+1}/{args.epochs_pre}  loss={total_loss/len(loader):.4f}')
            
            torch.cuda.empty_cache()

        if args.save_model:
            os.makedirs(args.output_path, exist_ok=True)
            torch.save(self.model.state_dict(),
                       os.path.join(args.output_path, 'pretrain.pt'))

    # ---------------------------------------------------------------
    # Steps 2 & 3: Iterative clustering + representation learning
    # ---------------------------------------------------------------
    def train(self, train_loader, test_loader):
        args = self.args
        ucl_loss = UnsupConLoss(tau=args.tau3).to(self.device)
        scl_loss = SupConLoss(tau=args.tau2).to(self.device)

        bert_params = list(self.model.bert.parameters())
        other_params = [p for n, p in self.model.named_parameters()
                        if 'bert' not in n]
        opt = AdamW([
            {'params': bert_params, 'lr': args.lr[0]},
            {'params': other_params, 'lr': args.lr[1]},
        ], weight_decay=1e-2)
        total_steps = args.epochs * len(train_loader)
        sched = get_linear_schedule_with_warmup(opt, int(total_steps * args.warmup), total_steps)

        best_avg = 0
        best_res = {}

        for ep in range(args.epochs):
            # -- Step 2: cluster and select high-quality samples --
            feats, y_true, _ = get_feats(self.model, train_loader, self.device)
            if self.centroids is None:
                preds, self.centroids = kmeans_pp(feats, self.k, seed=args.seed)
            else:
                preds, self.centroids = kmeans_pp(feats, self.k,
                                                   centroids=self.centroids, seed=args.seed)

            t = min(args.t0 + args.delta * ep, 1.0)
            hq = select_high_quality(feats, preds, self.k, t,
                                      L=0.1, delta_k=0.02, u=10)
            hq_idx = set(np.concatenate(list(hq.values())).tolist())

            # build pseudo-label array
            pseudo = np.full(len(preds), -1, dtype=int)
            for c, idxs in hq.items():
                pseudo[idxs] = c

            # -- Step 3: representation learning --
            self.model.train()
            total_loss = 0
            scaler = torch.cuda.amp.GradScaler()
            opt.zero_grad()
            for step, batch in enumerate(train_loader):
                ids = batch['text_ids'].to(self.device)
                mask = batch['text_mask'].to(self.device)
                seg = batch['text_seg'].to(self.device)
                v = batch['video'].to(self.device)
                a = batch['audio'].to(self.device)
                bidx = batch['idx'].numpy()

                with torch.cuda.amp.autocast():
                    ztav, zta0, zt0v = self.model.get_aug_reprs(ids, mask, seg, v, a)

                    # split high vs low quality in this batch
                    hq_mask = torch.tensor([i in hq_idx for i in bidx], dtype=torch.bool)
                    lq_mask = ~hq_mask

                    loss = torch.tensor(0.0, device=self.device)

                    # supervised contrastive on high-quality samples (all 3 views)
                    if hq_mask.any():
                        hi = hq_mask.to(self.device)
                        plabels = torch.tensor(pseudo[bidx], dtype=torch.long, device=self.device)
                        z_hq = torch.cat([ztav[hi], zta0[hi], zt0v[hi]], dim=0)
                        pl_hq = plabels[hi].repeat(3)
                        h_hq = self.model.head2(z_hq)
                        loss = loss + scl_loss(h_hq, pl_hq)

                    # unsupervised contrastive on low-quality samples
                    if lq_mask.any():
                        li = lq_mask.to(self.device)
                        z_lq = torch.cat([ztav[li], zta0[li], zt0v[li]], dim=0)
                        h_lq = self.model.head3(z_lq)
                        loss = loss + ucl_loss(h_lq)

                    loss = loss / args.grad_acc_steps

                scaler.scale(loss).backward()
                
                if (step + 1) % args.grad_acc_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                    opt.zero_grad()

                total_loss += loss.item() * args.grad_acc_steps

            torch.cuda.empty_cache()

            # evaluate every epoch
            res = self.evaluate(test_loader)
            avg = res['Avg']
            if avg > best_avg:
                best_avg = avg
                best_res = res
                if args.save_model:
                    os.makedirs(args.output_path, exist_ok=True)
                    torch.save(self.model.state_dict(),
                               os.path.join(args.output_path, 'best.pt'))

            print(f'[Train] Epoch {ep+1}/{args.epochs}  t={t:.2f}  loss={total_loss/len(train_loader):.4f}  '
                  f'NMI={res["NMI"]:.2f}  ARI={res["ARI"]:.2f}  ACC={res["ACC"]:.2f}  FMI={res["FMI"]:.2f}')

        print('\n=== Best Results ===')
        for k, v in best_res.items():
            print(f'  {k}: {v:.2f}')
        return best_res

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------
    def evaluate(self, loader):
        feats, y_true, _ = get_feats(self.model, loader, self.device)
        km = KMeans(n_clusters=self.k, init='k-means++' if self.centroids is None
                    else self.centroids, n_init=1 if self.centroids is not None else 10,
                    random_state=self.args.seed)
        preds = km.fit_predict(feats)
        return calc_metrics(y_true, preds)
