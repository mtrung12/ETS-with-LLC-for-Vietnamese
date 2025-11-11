import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from underthesea import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

class SummDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]['Text']
        summary = self.df.iloc[idx]['Summary']  # For eval only
        sentences = sent_tokenize(text)
        lengths = [len(word_tokenize(s)) for s in sentences]  # Syllable lengths
        return {'sentences': sentences, 'lengths': lengths, 'gold': summary}

def get_dataloaders(csv_path, batch_size=4):
    df = pd.read_csv(csv_path)
    train_df, temp_df = train_test_split(df, test_size=0.1, random_state=12)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=12)
    train_ds = SummDataset(train_df)
    val_ds = SummDataset(val_df)
    test_ds = SummDataset(test_df)
    collate_fn = lambda b: b  
    return (DataLoader(train_ds, batch_size, collate_fn=collate_fn),
            DataLoader(val_ds, batch_size, collate_fn=collate_fn),
            DataLoader(test_ds, batch_size, collate_fn=collate_fn))