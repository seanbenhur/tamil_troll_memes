from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
import torch

class MemeDataset(Dataset):
    def __init__(self,data_path,dir_path,transforms,train=True):
        self.df = pd.read_csv(data_path)
        
        self.dir_path = dir_path
        self.df['label'] = self.df['label'].replace(['not_troll','troll'],[0,1])
        self.transforms = transforms
        self.train = train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
   
        image_id = self.df['file_path'].values[idx]
        labels = self.df['label'].values[idx]
        image_path = os.path.join(self.dir_path,image_id)
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)
        label = torch.tensor(labels).float()
        return image, label
    
    

def create_dataset(
    dataset_path,
    transforms,
    dir_path
):
    train_ds = MemeDataset(
        dataset_path, transforms, dir_path
    )
    return train_ds


def create_dataloader(train_ds, eval_ds, batch_size):
    if eval_ds is None:
        train_dataloader = DataLoader(
            train_ds,
            batch_size=training_args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        return train_dataloader

    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataloader = DataLoader(
        eval_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader