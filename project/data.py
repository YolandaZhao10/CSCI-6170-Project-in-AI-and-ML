import os
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import Image

def get_base_transforms(name, is_train=True):
    """
    Creates dataset-specific transforms to normalize everything to
    1-channel 28x28 grayscale tensors.
    """
    t_list = []
    
    if name == 'SVHN':
        t_list.append(transforms.Grayscale(num_output_channels=1))
        t_list.append(transforms.Resize((28, 28)))
    elif name == 'USPS':
        # USPS is natively 16x16, so scale up
        t_list.append(transforms.Resize((28, 28)))
    
    if is_train:
        t_list.append(transforms.RandomRotation(15))
        t_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
        
    t_list.append(transforms.ToTensor())
    t_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    return transforms.Compose(t_list)

def load_dataset_split(name, root, train=True):
    transform = get_base_transforms(name, is_train=train)
    
    if name == 'MNIST':
        return datasets.MNIST(root, train=train, download=True, transform=transform)
    elif name == 'USPS':
        return datasets.USPS(root, train=train, download=True, transform=transform)
    elif name == 'SVHN':
        split = 'train' if train else 'test'
        return datasets.SVHN(root, split=split, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset {name}")

def get_multidomain_dataloaders(source_name, target_names, batch_size=64, download_dir='./data'):
    """
    Returns:
      - source_train_loader
      - source_test_loader
      - combined_target_train_loader
      - target_test_loaders_dict
    """
    source_train = load_dataset_split(source_name, download_dir, train=True)
    source_test = load_dataset_split(source_name, download_dir, train=False)

    source_train_loader = DataLoader(source_train, batch_size=batch_size, shuffle=True, drop_last=True)
    source_test_loader = DataLoader(source_test, batch_size=batch_size, shuffle=False)

    target_train_datasets = []
    target_test_loaders_dict = {}

    for tgt in target_names:
        t_train = load_dataset_split(tgt, download_dir, train=True)
        t_test = load_dataset_split(tgt, download_dir, train=False)
        
        target_train_datasets.append(t_train)
        target_test_loaders_dict[tgt] = DataLoader(t_test, batch_size=batch_size, shuffle=False)

    if len(target_train_datasets) > 1:
        combined_target_train = ConcatDataset(target_train_datasets)
    elif len(target_train_datasets) == 1:
        combined_target_train = target_train_datasets[0]
    else:
        combined_target_train = None

    if combined_target_train is not None:
        combined_target_train_loader = DataLoader(combined_target_train, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        combined_target_train_loader = None

    return source_train_loader, source_test_loader, combined_target_train_loader, target_test_loaders_dict
