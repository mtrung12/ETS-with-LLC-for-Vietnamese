import yaml
import torch.optim as optim
from models import LearnableSumm
from dataset import get_dataloaders
from trainer import train
import argparse  # Added for command-line args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs.yaml', help='Path to config file')
    parser.add_argument('--ratio', type=float, default=None, help='Optional compression ratio for post-train testing (0.4, 0.6, 0.8)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model = LearnableSumm(use_knapsack=config['model']['use_knapsack'])
    train_dl, _, _ = get_dataloaders(config['dataset']['path'], config['training']['batch_size'])
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    train(model, train_dl, optimizer, epochs=config['training']['epochs'], ratios=config['training']['ratios'])
    torch.save(model.state_dict(), 'models/model.pth')
    if args.ratio:
        print(f"Post-train: Using ratio {args.ratio} for quick test...")
        # Optional: Add quick eval here if needed