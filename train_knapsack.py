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
from torch.cuda.amp import autocast, GradScaler

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
    def __init__(self, num_samples=1_000_000, avg_sent=20, avg_len=25):
        self.budgets = [100, 150, 200, 250, 300]
        self.data = []
        for _ in range(num_samples):
            n = max(3, np.random.poisson(avg_sent))
            lengths = np.random.gamma(2, avg_len / 2, n).astype(np.float32)
            scores = np.random.uniform(0, 1, n).astype(np.float32)
            scores /= (scores.sum() + 1e-8)
            budget = random.choice(self.budgets)
            label = np.array(knapsack_dp(scores.tolist(), lengths.tolist(), budget), dtype=np.float32)
            self.data.append((scores, lengths, label))

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
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for scores, lengths, labels, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            scores = scores.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True).bool()

            optimizer.zero_grad()
            with autocast():
                outputs = model(scores, lengths)
                loss_per_elem = criterion(outputs, labels)
                loss = (loss_per_elem * masks).sum() / masks.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for scores, lengths, labels, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                scores = scores.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True).bool()

                with autocast():
                    outputs = model(scores, lengths)
                    loss_per_elem = criterion(outputs, labels)
                    loss = (loss_per_elem * masks).sum() / masks.sum()
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output_path)
            print(f"New best model saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1_000_000)
    parser.add_argument('--avg_sent', type=int, default=20)
    parser.add_argument('--avg_len', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--output_path', type=str, default='models/knapsack_pretrained.pth')
    args = parser.parse_args()
    train_knapsack(args)