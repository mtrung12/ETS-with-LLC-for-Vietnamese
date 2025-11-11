# evaluate.py
import argparse
import torch
from models import LearnableSumm
from dataset import get_dataloaders
from utils import extract_summary, compute_rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, required=True, choices=[0.4, 0.6, 0.8])
    args = parser.parse_args()

    _, _, test_dl = get_dataloaders()
    model = LearnableSumm(use_knapsack=True, knapsack_path='knapsack_pretrained.pth').to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    rouge_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    with torch.no_grad():
        for batch in test_dl:
            for item in batch:
                budget = args.ratio * item['total_syllables']
                pred = extract_summary(model.to('cpu'), item['sentences'], item['lengths'], budget)
                scores = compute_rouge(pred, item['gold'])
                for k in rouge_sum:
                    rouge_sum[k] += scores[k].fmeasure
    n = len(test_dl.dataset)
    print("ROUGE:", {k: round(v/n, 4) for k, v in rouge_sum.items()})