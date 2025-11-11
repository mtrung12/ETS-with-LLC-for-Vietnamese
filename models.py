import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class KnapsackTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=8):
        super().__init__()
        self.input_linear = nn.Linear(2, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        self.output_linear = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, scores, lengths):
        input = torch.stack([scores, lengths], dim=-1)  # (B, n, 2)
        input = self.input_linear(input)
        output = self.transformer(input)
        output = self.output_linear(output).squeeze(-1)
        return self.sigmoid(output)

class LearnableSumm(nn.Module):
    def __init__(self, use_knapsack=False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        self.encoder = AutoModel.from_pretrained('vinai/phobert-base')
        self.scorer = nn.Linear(768, 1)  # Score vi
        self.predictor = TransformerEncoder(TransformerEncoderLayer(768, 8), num_layers=2)
        self.knapsack = KnapsackTransformer() if use_knapsack else None
        self.use_knapsack = use_knapsack

    def encode_sentences(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        embeds = self.encoder(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
        return embeds

    def forward(self, sentences, lengths, budget=None):
        embeds = self.encode_sentences(sentences)  # (n, 768)
        scores = torch.sigmoid(self.scorer(embeds)).squeeze(-1)  # (n,)
        
        if self.use_knapsack and budget is not None:
            select_probs = self.knapsack(scores.unsqueeze(0), lengths.unsqueeze(0)).squeeze(0)
            select = torch.round(select_probs)  # For inference
            # Straight-through for training
            select_st = select - select_probs.detach() + select_probs
        else:
            _, topk_idx = torch.topk(scores, k=3, sorted=True)
            select_st = torch.zeros_like(scores)
            select_st[topk_idx] = 1.0
            select = select_st.clone().detach()

        # Mask for HS'', HS', HS
        HS = self.predictor(embeds.unsqueeze(0)).squeeze(0)  # Full doc
        HS_pp = HS * select_st.unsqueeze(-1)  # Extracted (HS'')
        # Random mask one selected for Rest-Ext
        selected_idx = torch.nonzero(select_st).squeeze(-1)
        if len(selected_idx) > 0:
            rand_idx = selected_idx[torch.randint(0, len(selected_idx), (1,))]
            mask = torch.ones_like(select_st)
            mask[rand_idx] = 0
            HS_p = HS * mask.unsqueeze(-1)  # Rest
        else:
            HS_p = HS

        # Predictions
        pred_ext = self.predictor(HS_p.unsqueeze(0)).squeeze(0)[rand_idx] if len(selected_idx) > 0 else torch.zeros(768)
        pred_doc = torch.mean(self.predictor(HS_pp.unsqueeze(0)).squeeze(0), dim=0)

        return scores, select, HS, pred_ext, pred_doc, rand_idx