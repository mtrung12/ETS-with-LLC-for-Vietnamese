import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import random
from tqdm import tqdm
from models import KnapsackTransformer
from utils import knapsack_dp
import argparse

class SyntheticKnapsackDataset(Dataset):
    def __init__(self, num_samples=6_000_000, avg_sent=20, avg_len=25):
        self.data = self.generate_data(num_samples, avg_sent, avg_len)

    def generate_data(self, num_samples, avg_sent, avg_len):
        data = []
        budgets = [100, 150, 200, 250, 300]  
        for _ in range(num_samples):
            n = max(3, np.random.poisson(avg_sent))
            lengths = np.random.gamma(2, avg_len / 2, n)
            scores = np.random.uniform(0, 1, n)
            scores /= scores.sum()
            budget = random.choice(budgets)
            label = knapsack_dp(scores.tolist(), lengths.tolist(), budget)
            data.append((scores, lengths, label, budget))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scores, lengths, label, _ = self.data[idx]
        return (
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(lengths, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

def train_knapsack(args):
    print("Generating synthetic data...")
    dataset = SyntheticKnapsackDataset(
        num_samples=args.num_samples,
        avg_sent=args.avg_sent,
        avg_len=args.avg_len
    )
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = KnapsackTransformer(d_model=768, nhead=8, num_layers=8)
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for scores, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            optimizer.zero_grad()
            outputs = model(scores, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for scores, lengths, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                outputs = model(scores, lengths)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), args.output_path)
            print(f"New best model saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=1_000_000, help='Số mẫu synthetic (giảm để test nhanh)')
    parser.add_argument('--avg_sent', type=int, default=20, help='Số câu trung bình')
    parser.add_argument('--avg_len', type=int, default=25, help='Độ dài câu trung bình (syllables)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--output_path', type=str, default='models/knapsack_pretrained.pth')
    args = parser.parse_args()
    train_knapsack(args)