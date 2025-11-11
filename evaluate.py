# evaluate.py
import argparse
import yaml
import torch
from models import LearnableSumm
from dataset import get_dataloaders
from utils import extract_summary, compute_rouge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs.yaml', help='Path to config file')
    parser.add_argument('--ratio', type=float, required=True, choices=[0.4, 0.6, 0.8])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    csv_path = cfg['dataset']['path']
    batch_size = cfg['evaluation']['batch_size'] 
    # Dataloader
    _, _, test_dl = get_dataloaders(csv_path=csv_path, batch_size=batch_size)

    # Model
    model = LearnableSumm(
        use_knapsack=cfg['model']['use_knapsack'],
        knapsack_path=cfg['model']['knapsack_path']
    ).to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    rouge_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    with torch.no_grad():
        for batch in test_dl:
            for item in batch:
                budget = args.ratio * item['total_syllables']
                pred = extract_summary(model, item['sentences'], item['lengths'], budget)
                scores = compute_rouge(pred, item['gold'])
                for k in rouge_sum:
                    rouge_sum[k] += scores[k].fmeasure

    n = len(test_dl.dataset)
    result = {k: round(v / n, 4) for k, v in rouge_sum.items()}
    print("ROUGE:", result)