import argparse  # Added for command-line args
from models import LearnableSumm
from dataset import get_dataloaders
from utils import extract_summary, compute_rouge
import torch 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float, default=0.6, choices=[0.4, 0.6, 0.8], help='Compression ratio: 0.4, 0.6, or 0.8')
    args = parser.parse_args()

    _, _, test_dl = get_dataloaders('data/dataset.csv')
    model = LearnableSumm(use_knapsack=True)
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()
    total_rouge = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for batch in test_dl:
        for item in batch:
            sentences = item['sentences']
            lengths = item['lengths']
            total_syllables = sum(lengths)  
            budget = args.ratio * total_syllables  
            pred = extract_summary(model, sentences, lengths, budget)
            scores = compute_rouge(pred, item['gold'])
            for k in total_rouge:
                total_rouge[k] += scores[k].fmeasure
    print({k: v / len(test_dl) for k, v in total_rouge.items()})