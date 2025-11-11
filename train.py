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
    batch_size = HS.shape[0]

    # === L_rest: Rest-Ext ===
    # pred_ext: [batch, max_sents, 768], chỉ lấy tại rand_idx
    valid_rand = (rand_idx >= 0) & mask.any(dim=1)
    if valid_rand.any():
        pred_ext_valid = pred_ext[valid_rand][torch.arange(valid_rand.sum()), rand_idx[valid_rand]]
        HS_at_rand = HS[valid_rand][torch.arange(valid_rand.sum()), rand_idx[valid_rand]]
        pos_cos = F.cosine_similarity(pred_ext_valid, HS_at_rand, dim=1)
    else:
        pos_cos = torch.tensor([], device=HS.device)

    # Negative: random unselected
    neg_cos_list = []
    for i in range(batch_size):
        if not valid_rand[i]: continue
        unselected = (select[i] == 0) & mask[i]
        if unselected.any():
            neg_i = unselected.nonzero(as_tuple=True)[0]
            neg_j = neg_i[torch.randint(0, len(neg_i), (1,))]
            neg_cos_list.append(F.cosine_similarity(pred_ext[i, rand_idx[i]], HS[i, neg_j], dim=0))
    neg_cos = torch.stack(neg_cos_list) if neg_cos_list else torch.tensor([], device=HS.device)

    L_rest = (-pos_cos + neg_cos.abs()).mean() if len(pos_cos) > 0 else torch.tensor(0.0, device=HS.device)

    # === L_doc: Ext-Doc ===
    true_doc = (HS * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

    # pred_doc là mean của selected → đã tính đúng
    pos_doc = F.cosine_similarity(pred_doc, true_doc, dim=1)

    # neg_doc: mean của unselected
    unselected_mask = (1 - select) * mask
    has_unselected = unselected_mask.sum(1) > 0
    neg_doc = torch.zeros_like(true_doc)
    if has_unselected.any():
        neg_doc[has_unselected] = (HS[has_unselected] * unselected_mask[has_unselected].unsqueeze(-1)).sum(1) / unselected_mask[has_unselected].sum(1, keepdim=True)

    neg_doc_cos = F.cosine_similarity(neg_doc, true_doc, dim=1).abs()
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
        knapsack_path=cfg["model"]["knapsack_path"]
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
                        s, sel, h, pe, pd, ri = model(sents, batch['lengths_list'][i], budget.item())   
                    scores.append(s); select.append(sel); HS.append(h); pred_ext.append(pe); pred_doc.append(pd); rand_idx.append(ri)

                max_sents = max(h.shape[0] for h in HS)
                batch_size = len(HS)

                HS_padded = torch.zeros(batch_size, max_sents, 768, device=device)
                pred_ext_padded = torch.zeros(batch_size, max_sents, 768, device=device)
                pred_doc_padded = torch.zeros(batch_size, 768, device=device)  # pred_doc là mean → shape [768]
                select_padded = torch.zeros(batch_size, max_sents, device=device)
                rand_idx_padded = torch.full((batch_size,), -1, dtype=torch.long, device=device)  # -1 nếu không có

                for i, (h, pe, pd, sel, ri) in enumerate(zip(HS, pred_ext, pred_doc, select, rand_idx)):
                    n = h.shape[0]
                    HS_padded[i, :n] = h
                    pred_ext_padded[i, :n] = pe
                    pred_doc_padded[i] = pd
                    select_padded[i, :n] = sel
                    if ri.item() != 0 or n > 0:  
                        rand_idx_padded[i] = ri

                mask = batch['lengths_mask'].to(device).bool()  # [batch_size, max_sents]

                HS = HS_padded
                pred_ext = pred_ext_padded
                pred_doc = pred_doc_padded
                select = select_padded
                rand_idx = rand_idx_padded

                loss = compute_loss_batch(HS, pred_ext, pred_doc, rand_idx, select, lengths_mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(train_dl):.4f}")
    torch.save(model.state_dict(), 'model.pth')