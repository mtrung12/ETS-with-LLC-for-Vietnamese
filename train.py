import yaml
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from models import LearnableSumm
from dataset import get_dataloaders
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_loss_batch(HS, pred_ext, pred_doc, rand_idx, select, mask):
    mask = mask.bool()
    HS = HS * mask.unsqueeze(-1)
    select = select * mask

    # L_rest: Rest-Ext
    pos_cos = torch.cosine_similarity(pred_ext, HS[range(len(rand_idx)), rand_idx], dim=1)
    unselected = (select == 0) & mask
    neg_idx = []
    for i in range(len(unselected)):
        idxs = torch.nonzero(unselected[i]).squeeze()
        neg_idx.append(idxs[torch.randint(0, len(idxs), (1,))] if len(idxs) > 0 else 0)
    neg_idx = torch.stack(neg_idx)
    neg_cos = torch.cosine_similarity(pred_ext, HS[range(len(neg_idx)), neg_idx], dim=1).abs()
    L_rest = (-pos_cos + neg_cos).mean()

    # L_doc: Ext-Doc
    true_doc = (HS * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    pred_doc_mean = pred_doc.mean(1)
    pos_doc = torch.cosine_similarity(pred_doc_mean, true_doc, dim=1)
    neg_doc = (HS * (1 - select).unsqueeze(-1) * mask.unsqueeze(-1)).sum(1) / (mask.sum(1, keepdim=True) - select.sum(1, keepdim=True))
    neg_doc_cos = torch.cosine_similarity(neg_doc, true_doc, dim=1).abs()
    L_doc = (-pos_doc + neg_doc_cos).mean()

    return L_rest + L_doc

if __name__ == '__main__':
    with open('configs.yaml') as f:
        cfg = yaml.safe_load(f)

    train_dl, _, _ = get_dataloaders(
        csv_path=cfg['dataset']['path'],
        batch_size=cfg['training']['batch_size']
    )
    model = LearnableSumm(
        use_knapsack=cfg['model']['use_knapsack'],
        knapsack_path=config["model"]["knapsack_path"]
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    scaler = GradScaler()

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            sentences_list = batch['sentences_list']
            lengths = batch['lengths'].to(device)
            lengths_mask = batch['lengths_mask'].to(device)
            total_syllables = batch['total_syllables'].to(device)
            ratios = random.choices(cfg['training']['ratios'], k=len(total_syllables))
            budgets = [r * t for r, t in zip(ratios, total_syllables)]

            with autocast():
                scores, select, HS, pred_ext, pred_doc, rand_idx = [], [], [], [], [], []
                for i, (sents, budget) in enumerate(zip(sentences_list, budgets)):
                    with torch.no_grad():
                        s, sel, h, pe, pd, ri = model(sents, lengths[i].tolist(), budget.item())
                    scores.append(s); select.append(sel); HS.append(h); pred_ext.append(pe); pred_doc.append(pd); rand_idx.append(ri)

                HS = torch.stack(HS).to(device)
                pred_ext = torch.stack(pred_ext).to(device)
                pred_doc = torch.stack(pred_doc).to(device)
                select = torch.stack(select).to(device)
                rand_idx = torch.stack(rand_idx).to(device)

                loss = compute_loss_batch(HS, pred_ext, pred_doc, rand_idx, select, lengths_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(train_dl):.4f}")
    torch.save(model.state_dict(), 'model.pth')