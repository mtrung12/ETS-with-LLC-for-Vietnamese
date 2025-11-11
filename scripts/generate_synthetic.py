import numpy as np
import torch
from src.utils import knapsack_dp
from src.models import KnapsackTransformer
from torch.utils.data import TensorDataset, DataLoader

def generate_synthetic(num_samples=6000000, avg_sent=20, avg_len=20):
    # Poisson for num sents, Gamma for lens, Uniform scores
    num_sents = np.random.poisson(avg_sent, num_samples)
    data = []
    for ns in num_sents:
        lens = np.random.gamma(2, avg_len/2, ns)
        scores = np.random.uniform(0, 1, ns)
        scores /= scores.sum()
        lens /= lens.max()  # Normalize
        budgets = np.random.choice([100, 150, 200, 250, 300], 1)[0]
        labels = knapsack_dp(scores, lens, budgets)
        data.append((scores, lens, labels))
    return data

if __name__ == '__main__':
    data = generate_synthetic()
    # Split 95/5
    train_data = data[:int(0.95 * len(data))]
    val_data = data[int(0.95 * len(data)):]
    # Train Knapsack
    model = KnapsackTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()
    train_ds = TensorDataset(torch.tensor([d[0] for d in train_data]), torch.tensor([d[1] for d in train_data]), torch.tensor([d[2] for d in train_data]))
    train_dl = DataLoader(train_ds, batch_size=32)
    for epoch in range(10):
        for scores, lens, labels in train_dl:
            optimizer.zero_grad()
            preds = model(scores, lens)
            loss = criterion(preds, labels.float())
            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'models/knapsack.pth')
