import os

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import argparse

import torch

from config import param_map
from dataloader import get_loaders
from manager import UMCManager
from metrics import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        default="MIntRec",
        choices=["MIntRec", "MELD-DA", "IEMOCAP-DA"],
    )
    p.add_argument("--data_path", type=str, default="Datasets")
    p.add_argument("--bert_path", type=str, default="bert-base-uncased")
    p.add_argument("--output_path", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pretrain", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument(
        "--pretrain_path", type=str, default=None, help="Path to saved pretrain weights"
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_acc_steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = param_map[args.dataset]
    for k, v in cfg.items():
        if k == "batch" and args.batch_size is not None:
            continue
        if k == "grad_acc_steps" and args.grad_acc_steps is not None:
            continue
        setattr(args, k, v)
    if args.batch_size is not None:
        args.batch = args.batch_size
    if not hasattr(args, "grad_acc_steps"):
        args.grad_acc_steps = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}  |  K={args.num_labels}")

    loaders = get_loaders(
        data_path=args.data_path,
        dataset=args.dataset,
        bert_path=args.bert_path,
        seq_lens=args.seq_len,
        batch_size=args.batch,
        seed=args.seed,
    )
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    mgr = UMCManager(args, device)

    # Step 1: pre-training
    if args.pretrain:
        print("\n--- Pre-training ---")
        mgr.pretrain(train_loader)
    elif args.pretrain_path:
        print(f"Loading pretrain weights from {args.pretrain_path}")
        mgr.model.load_state_dict(torch.load(args.pretrain_path, map_location=device))

    # Steps 2 & 3: iterative clustering + representation learning
    if args.train:
        print("\n--- Training ---")
        results = mgr.train(train_loader, test_loader)
    else:
        print("\n--- Evaluation only ---")
        results = mgr.evaluate(test_loader)
        print("Results:", results)


if __name__ == "__main__":
    main()
