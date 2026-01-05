import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .dataset import FERDataset


def get_data_loaders(
    data_dir="./data/fer2013_clean.csv", batch_size=32, num_workers=4, oversample=False
):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    val_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    train_dataset = FERDataset(
        filename=data_dir, transform=train_transform, usage="Training"
    )
    val_dataset = FERDataset(
        filename=data_dir, transform=val_transform, usage="PublicTest"
    )
    test_dataset = FERDataset(
        filename=data_dir, transform=val_transform, usage="PrivateTest"
    )

    if oversample:
        sample_weights = train_dataset.get_sample_weights()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
