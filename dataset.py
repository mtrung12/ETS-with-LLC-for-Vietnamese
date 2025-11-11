import torch
from torch.utils.data import Dataset, DataLoader
from underthesea import sent_tokenize, word_tokenize
import pandas as pd
from sklearn.model_selection import train_test_split

class SummDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentences = sent_tokenize(row['Text'])
        lengths = [len(word_tokenize(s)) for s in sentences]
        return {
            'sentences': sentences,
            'lengths': lengths,
            'total_syllables': sum(lengths),
            'gold': row['Summary']
        }

def pad_sequence(seqs, padding_value=0):
    max_len = max(len(s) for s in seqs)
    padded = []
    masks = []
    for s in seqs:
        pad = max_len - len(s)
        padded.append(s + [padding_value] * pad)
        masks.append([1] * len(s) + [0] * pad)
    return torch.tensor(padded), torch.tensor(masks)

def collate_fn(batch):
    sentences_list = [item['sentences'] for item in batch]
    lengths_list = [item['lengths'] for item in batch]
    total_syllables = torch.tensor([item['total_syllables'] for item in batch])
    gold_list = [item['gold'] for item in batch]

    lengths_padded, lengths_mask = pad_sequence(lengths_list, 0)
    total_syllables = total_syllables.float()

    return {
        'sentences_list': sentences_list,
        'lengths': lengths_padded,
        'lengths_mask': lengths_mask,
        'total_syllables': total_syllables,
        'gold_list': gold_list
    }

def get_dataloaders(csv_path='dataset.csv', batch_size=64):
    df = pd.read_csv(csv_path)
    train_df, temp = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=42)
    return (
        DataLoader(SummDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(SummDataset(val_df), batch_size=batch_size, collate_fn=collate_fn),
        DataLoader(SummDataset(test_df), batch_size=batch_size, collate_fn=collate_fn)
    )