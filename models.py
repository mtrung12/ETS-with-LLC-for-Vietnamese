# models.py
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KnapsackTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=8):
        super().__init__()
        self.input_linear = nn.Linear(2, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.output_linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores, lengths):
        x = torch.stack([scores, lengths], dim=-1)
        x = self.input_linear(x)
        x = self.transformer(x)
        return self.sigmoid(self.output_linear(x).squeeze(-1))

class LearnableSumm(nn.Module):
    def __init__(self, use_knapsack=False, knapsack_path=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.encoder = AutoModel.from_pretrained('vinai/phobert-base').to(device)
        self.scorer = nn.Linear(768, 1).to(device)
        self.predictor = TransformerEncoder(TransformerEncoderLayer(768, 8, batch_first=True), num_layers=2).to(device)
        self.use_knapsack = use_knapsack
        if use_knapsack:
            self.knapsack = KnapsackTransformer().to(device)
            if knapsack_path and os.path.exists(knapsack_path):
                self.knapsack.load_state_dict(torch.load(knapsack_path, map_location=device))
                print(f"Knapsack loaded on {device}")

    def encode_sentences(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
        with torch.no_grad():
            return self.encoder(**inputs).last_hidden_state[:, 0, :]

    def forward(self, sentences, lengths, budget=None):
        embeds = self.encode_sentences(sentences)
        scores = torch.sigmoid(self.scorer(embeds)).squeeze(-1)

        if self.use_knapsack and budget is not None:
            lengths = torch.tensor(lengths, dtype=torch.float, device=device)
            select_probs = self.knapsack(scores.unsqueeze(0), lengths.unsqueeze(0)).squeeze(0)
            select = torch.round(select_probs)
            select_st = select - select_probs.detach() + select_probs
        else:
            _, topk = torch.topk(scores, k=3)
            select_st = torch.zeros_like(scores)
            select_st[topk] = 1.0
            select = select_st.clone()

        HS = self.predictor(embeds.unsqueeze(0)).squeeze(0)
        HS_pp = HS * select_st.unsqueeze(-1)
        selected_idx = torch.nonzero(select_st).squeeze()
        if len(selected_idx) > 0:
            rand_idx = selected_idx[torch.randint(0, len(selected_idx), (1,))]
            mask = torch.ones_like(select_st); mask[rand_idx] = 0
            HS_p = HS * mask.unsqueeze(-1)
        else:
            HS_p = HS; rand_idx = torch.tensor(0)

        pred_ext = self.predictor(HS_p.unsqueeze(0)).squeeze(0)[rand_idx]
        pred_doc = torch.mean(self.predictor(HS_pp.unsqueeze(0)).squeeze(0), dim=0)

        return scores.cpu(), select.cpu(), HS.cpu(), pred_ext.cpu(), pred_doc.cpu(), rand_idx