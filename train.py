# train.py
import yaml
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from models import LearnableSumm
from dataset import get_dataloaders
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

def compute_loss(HS, pred_ext, pred_doc, rand_idx, select):
    pos_cos = torch.cosine_similarity(pred_ext, HS[rand_idx], dim=0)
    unselected = torch.nonzero(select == 0).squeeze()
    neg_idx = unselected[torch.randint(0, len(unselected), (1,))] if len(unselected) > 0 else 0
    neg_cos = torch.cosine_similarity(pred_ext, HS[neg_idx], dim=0).abs()
    L_rest = -pos_cos + neg_cos

    true_doc = torch.mean(HS, dim=0)
    pos_doc = torch.cosine_similarity(pred_doc, true_doc, dim=0)
    neg_doc = torch.mean(HS * (1 - select).unsqueeze(-1), dim=0)
    neg_doc_cos = torch.cosine_similarity(neg_doc, true_doc, dim=0).abs()
    L_doc = -pos_doc + neg_doc_cos
    return L_rest + L_doc

if __name__ == '__main__':
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    model = LearnableSumm(
        use_knapsack=cfg['model']['use_knapsack'],
        knapsack_path='knapsack_pretrained.pth'
    ).to(device)

    train_dl, _, _ = get_dataloaders(cfg['training']['batch_size'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    scaler = GradScaler()  # Mixed precision

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            for item in batch:
                sentences = item['sentences']
                lengths = [int(l) for l in item['lengths']]
                total = item['total_syllables']
                ratio = random.choice(cfg['training']['ratios'])
                budget = ratio * total

                with autocast():
                    scores, select, HS, pred_ext, pred_doc, rand_idx = model(sentences, lengths, budget)
                    HS, pred_ext, pred_doc = HS.to(device), pred_ext.to(device), pred_doc.to(device)
                    select = select.to(device)
                    loss = compute_loss(HS, pred_ext, pred_doc, rand_idx, select)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
        print(f"Loss: {total_loss/len(train_dl):.4f}")
    torch.save(model.state_dict(), 'model.pth')