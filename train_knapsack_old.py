# train_knapsack.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
from models import KnapsackTransformer
from utils import knapsack_dp
import argparse
import os
from torch.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def knapsack_collate_fn(batch):
    scores, lengths, labels = zip(*batch)
    max_n = max(s.shape[0] for s in scores)
    padded_scores = []
    padded_lengths = []
    padded_labels = []
    masks = []

    for s, l, lbl in zip(scores, lengths, labels):
        n = s.shape[0]
        pad = max_n - n
        if pad > 0:
            s = torch.cat([s, torch.zeros(pad, dtype=s.dtype)])
            l = torch.cat([l, torch.zeros(pad, dtype=l.dtype)])
            lbl = torch.cat([lbl, torch.zeros(pad, dtype=lbl.dtype)])
        mask = torch.cat([torch.ones(n), torch.zeros(pad)])
        padded_scores.append(s)
        padded_lengths.append(l)
        padded_labels.append(lbl)
        masks.append(mask)

    return (
        torch.stack(padded_scores),
        torch.stack(padded_lengths),
        torch.stack(padded_labels),
        torch.stack(masks)
    )


class SyntheticKnapsackDataset(Dataset):
    def __init__(self, num_samples=1_000_000, avg_sent=17, avg_len=23):
        self.data = []
        self.ratios = [0.4, 0.6, 0.8] 

        print(f"Generating {num_samples:,} synthetic samples (Sent-512 ≈ {avg_sent}, avg_len ≈ {avg_len})...")

        for i in range(num_samples):
            if i % 500_000 == 0 and i > 0:
                print(f"  → {i:,}/{num_samples:,}")

            # Số câu: Poisson(Sent-512)
            n = max(3, np.random.poisson(avg_sent))

            # Độ dài: Gamma(α=2, β=avg_len/2)
            lengths = np.random.gamma(2, avg_len / 2, n).astype(np.float32)

            # Score: Uniform(0,1) → normalize
            scores = np.random.uniform(0, 1, n).astype(np.float32)
            scores /= (scores.sum() + 1e-8)

            # Budget: ratio × total_length
            ratio = random.choice(self.ratios)
            budget = int(ratio * lengths.sum())

            # DP Label
            label = np.array(knapsack_dp(scores.tolist(), lengths.tolist(), budget), dtype=np.float32)
            self.data.append((scores, lengths, label))

        print("Generation completed.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s, l, lbl = self.data[idx]
        return (
            torch.from_numpy(s),
            torch.from_numpy(l),
            torch.from_numpy(lbl)
        )


def train_knapsack(args):
    print(f"Using device: {device}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    dataset = SyntheticKnapsackDataset(
        num_samples=args.num_samples,
        avg_sent=args.avg_sent,
        avg_len=args.avg_len
    )

    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        collate_fn=knapsack_collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=2,
        collate_fn=knapsack_collate_fn
    )

    model = KnapsackTransformer(d_model=768, nhead=8, num_layers=8).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_matched_rate = 0.0

    for epoch in range(args.epochs):
        # =================== TRAINING ===================
        model.train()
        train_loss = 0.0
        for scores, lengths, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            scores = scores.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).bool()

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(scores, lengths)
                loss_per_elem = criterion(logits, labels)
                loss = (loss_per_elem * masks).sum() / masks.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # =================== VALIDATION ===================
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for scores, lengths, labels, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                scores = scores.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).bool()

                with autocast('cuda'):
                    logits = model(scores, lengths)
                    loss_per_elem = criterion(logits, labels)
                    loss = (loss_per_elem * masks).sum() / masks.sum()
                val_loss += loss.item()

                # Tính Matched Rate
                pred = (logits.sigmoid() > 0.5).float()
                pred_masked = pred * masks
                label_masked = labels * masks

                for i in range(pred.shape[0]):
                    p = pred_masked[i][masks[i]].cpu().numpy()
                    l = label_masked[i][masks[i]].cpu().numpy()
                    if len(p) > 0 and np.array_equal(p, l):
                        correct += 1
                    if len(p) > 0:
                        total += 1

        avg_val = val_loss / len(val_loader)
        matched_rate = correct / total if total > 0 else 0

        print(f"Epoch {epoch+1}: "
              f"Train Loss = {avg_train:.6f}, "
              f"Val Loss = {avg_val:.6f}, "
              f"Val Matched Rate = {matched_rate:.4%}")

        # Lưu model tốt nhất theo Matched Rate
        if matched_rate > best_matched_rate:
            best_matched_rate = matched_rate
            best_path = args.output_path.replace('.pth', '_best_match.pth')
            torch.save(model.state_dict(), best_path)
            print(f"New best model (Matched Rate: {matched_rate:.4%}) saved to {best_path}")

        # Lưu model tốt nhất theo Val Loss
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output_path)
            print(f"New best model (Val Loss: {avg_val:.6f}) saved to {args.output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1_000_000, help='Number of synthetic samples')
    parser.add_argument('--avg_sent', type=int, default=17, help='Sent-512: avg sentences in 512 tokens')
    parser.add_argument('--avg_len', type=int, default=23, help='Average sentence length (words)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_path', type=str, default='models/knapsack_pretrained.pth')
    args = parser.parse_args()

    train_knapsack(args)