import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import FERDataset


def get_data_loaders(data_dir="./data", batch_size=32, num_workers=4):
    """Create data loaders with augmentations"""

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    train_dataset = FERDataset(
        data_dir=data_dir, transform=train_transform, usage="Training"
    )
    val_dataset = FERDataset(
        data_dir=data_dir, transform=val_transform, usage="PublicTest"
    )
    test_dataset = FERDataset(
        data_dir=data_dir, transform=val_transform, usage="PrivateTest"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

