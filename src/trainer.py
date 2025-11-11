import torch
import torch.nn.functional as F
from tqdm import tqdm
import random  # Added for random ratio

def compute_loss(HS, pred_ext, pred_doc, rand_idx, select):  
    pos_cos = F.cosine_similarity(pred_ext, HS[rand_idx], dim=0)
    unselected_idx = torch.nonzero(select == 0).squeeze(-1)
    if len(unselected_idx) > 0:
        neg_idx = unselected_idx[torch.randint(0, len(unselected_idx), (1,))]
        neg_cos = F.cosine_similarity(pred_ext, HS[neg_idx], dim=0).abs()
    else:
        neg_cos = 0.0
    L_rest_ext = -pos_cos + neg_cos

    true_doc = torch.mean(HS, dim=0)
    pos_doc_cos = F.cosine_similarity(pred_doc, true_doc, dim=0)
    neg_doc = torch.mean(HS * (1 - select).unsqueeze(-1), dim=0)
    neg_doc_cos = F.cosine_similarity(neg_doc, true_doc, dim=0).abs()
    L_ext_doc = -pos_doc_cos + neg_doc_cos

    return L_rest_ext + L_ext_doc  

def train(model, dataloader, optimizer, epochs=10, device='cuda', ratios=[0.4, 0.6, 0.8]):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            for item in batch:
                sentences = item['sentences']
                lengths = torch.tensor(item['lengths'], dtype=torch.float)
                total_syllables = sum(item['lengths'])  # Total doc length in syllables
                ratio = random.choice(ratios)  # Random compression ratio from list
                budget = ratio * total_syllables  # Compute budget based on ratio
                scores, select, HS, pred_ext, pred_doc, rand_idx = model(sentences, lengths, budget)
                loss = compute_loss(HS, pred_ext, pred_doc, rand_idx, select)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f'Epoch {epoch}: Loss {total_loss / len(dataloader)}')