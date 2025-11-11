from rouge_score import rouge_scorer
import torch 

def extract_summary(model, sentences, lengths, budget=None):
    lengths = torch.tensor(lengths, dtype=torch.float)  # Ensure tensor
    with torch.no_grad():
        scores, select, _, _, _, _ = model(sentences, lengths, budget)
        selected_idx = torch.nonzero(select).squeeze(-1)
        return ' '.join([sentences[i] for i in selected_idx])

def compute_rouge(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(gold, pred)

# DP for knapsack labels
def knapsack_dp(values, weights, capacity):
    capacity = int(capacity) + 1  # DP table size
    n = len(values)
    dp = [[0.0] * capacity for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - int(weights[i-1])] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]
    # Backtrack
    selected = [0] * n
    w = capacity - 1
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected[i-1] = 1
            w -= int(weights[i-1])
    return selected