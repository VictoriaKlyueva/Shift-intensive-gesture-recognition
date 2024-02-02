import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class SignDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].to_numpy().reshape(1, 28, 28)
        label = self.data.iloc[idx]['label']
        return torch.tensor(image).float(), torch.tensor(label)


def get_sign_dataloader(csv_path_train, csv_path_val, batch_size, shuffle=True, num_workers=1):
    transform_train = transforms.Compose([
        transforms.Normalize(mean=[0.5125, 0.4667, 0.4110],
                             std=[0.2621, 0.2501, 0.2453]),
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.RandomApply([transforms.RandomRotation(degrees=(-90, 90))], p=0.2),
    ])

    train_dataset = SignDataset(csv_file=csv_path_train, transform=transform_train)
    val_dataset = SignDataset(csv_file=csv_path_val)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }

    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        csv_path_test, batch_size, num_workers=1,
    ):
    test_dataset = SignDataset(csv_file=csv_path_test)

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
