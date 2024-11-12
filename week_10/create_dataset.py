#!/usr/bin/env python
# coding: utf-8
import os
import random

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset, TensorDataset, DataLoader
from torchvision.models import MobileNet_V2_Weights, SqueezeNet1_0_Weights, ResNeXt50_32X4D_Weights
from tqdm import tqdm
from tqdm.notebook import tqdm


# In[2]:
def sample_cifar_evenly(dataset, num_per_class, num_classes=10):
    """
    Samples images evenly across the classes in the CIFAR dataset.
    
    Parameters:
    - dataset: The CIFAR dataset (e.g., CIFAR10)
    - num_per_class: Number of images to sample per class
    
    Returns:
    - A Subset containing the sampled images
    """
    # Ensure the number of images requested per class does not exceed the number available
    indices_per_class = {i: [] for i in range(num_classes)}

    # Collect indices for each class
    for idx, (image, label) in tqdm(enumerate(dataset), desc="Sampling Subset", total=len(dataset)):
        if len(indices_per_class[label]) < num_per_class:
            indices_per_class[label].append(idx)
        if all([num_per_class == len(idxs) for idxs in indices_per_class.values()]):
            break

    # Sample indices from each class
    sampled_indices = []
    for label, indices in indices_per_class.items():
        if len(indices) >= num_per_class:
            sampled_indices.extend(random.sample(indices, num_per_class))
        else:
            print(f"Warning: Only {len(indices)} available for class {label}. Sampling all.")

    # Create a subset with the sampled indices
    print(f"Sampled {len(sampled_indices)} points")
    return Subset(dataset, sampled_indices)

# In[6]:


def cossim(img1, img2):
    return nn.CosineSimilarity(dim=1)(img1, img2)


# In[7]:


def get_relative_representations(dataset, anchors_embedded, model, filename_base, batch_size=32, use_own_reps=False):
    data, targets = [], []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    continue_counter = 0
    for batch_idx, (imgs, targets_) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Reps"):
        filename_batch = filename_base.replace("batch", f"batch_{batch_idx + 1}")

        if os.path.exists(filename_batch):
            print(f"Skipped {filename_batch}")
            continue_counter = 10
        continue_counter -= 0 < continue_counter

        if continue_counter:
            continue

        imgs = imgs.to(device)
        imgs_embedded = model.features(imgs) if "resnet" not in filename_base else model(imgs)
        # Process each image in the batch
        for img, target in zip(imgs_embedded, targets_):
            if not use_own_reps:
                repr = cossim(img.unsqueeze(0).reshape((1, -1)), anchors_embedded.reshape((len(anchors_embedded), -1)))
            else:
                repr = img.flatten()
            data.append(repr)
            targets.append(target)

        # Save after every batch
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:  # Save every 10 batches (or the last batch)
            data_batch = torch.vstack(data)
            targets_batch = torch.vstack(targets)
            dataset_batch = TensorDataset(data_batch, targets_batch)
            torch.save(dataset_batch, filename_batch)
            data, targets = [], []  # Reset the lists for the next batch

    # Final saving if any data is left to save
    if data:
        data_batch = torch.vstack(data)
        targets_batch = torch.vstack(targets)
        dataset_batch = TensorDataset(data_batch, targets_batch)
        torch.save(dataset_batch, filename_base.replace("batch", "final"))

    return data, targets


def load_and_concatenate_batches(folder, batch_size=32):
    """
    Load and concatenate data from multiple batch files, given a pattern.

    Args:
        file_pattern (str): The pattern for the saved .pth files (e.g., "data/rel_reps_mobilenet/train/batch_*.pth").
        batch_size (int): The batch size to use for loading.

    Returns:
        DataLoader: A DataLoader with the concatenated dataset.
    """
    # Find all batch files matching the pattern
    try:
        batch_files = os.listdir(folder)
    except FileNotFoundError:
        print(f"No data for {folder}")
        return None
    # Initialize lists to accumulate the data
    all_data, all_targets = [], []

    # Iterate through each batch file
    for batch_file in batch_files:
        # Load the TensorDataset for the current batch
        file_path = os.path.join(os.path.dirname(folder), batch_file)

        if os.path.exists(file_path):
            dataset = torch.load(file_path)
            # Accumulate the data and targets
            for data_batch, target_batch in dataset:
                all_data.append(data_batch)
                all_targets.append(target_batch)
        else:
            print(f"File not found: {file_path}")
    if not all_data:
        print(f"No data for {folder}")
        return None

    # Concatenate all batches into a single tensor
    all_data = torch.vstack(all_data).detach()
    all_targets = torch.cat(all_targets, dim=0).detach()
    # Create a TensorDataset and return a DataLoader
    full_dataset = TensorDataset(all_data, all_targets)
    return DataLoader(full_dataset, batch_size=batch_size, shuffle="train" in folder.lower())


if __name__ == "__main__":
    device = "cpu"  # It uses a lot of memory and does not have to be really fast
    print(device)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to 256 pixels on the smaller side
        transforms.CenterCrop(64),  # Crop to 224 x 224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Download CIFAR-10
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split the training data into training and validation sets
    train_size = int(0.8 * len(cifar10_train))  # 80% for training
    val_size = len(cifar10_train) - train_size  # Remaining 20% for validation
    cifar10_train, cifar10_val = random_split(cifar10_train, [train_size, val_size])

    mobilenet = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(device)
    squeezenet = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).to(device)
    resnet = models.resnet50(weights=ResNeXt50_32X4D_Weights)
    modules = list(resnet.children())[:-2]  # Remove avgpool and fc
    resnet = torch.nn.Sequential(*modules)

    mobilenet.eval()
    squeezenet.eval()
    resnet.eval()


    # In[16]:
    anchor_subset = sample_cifar_evenly(cifar10_train, num_per_class=30)


    # In[17]:
    # Generate relative representations for the train, validation, and test sets
    anchor_subset_matrix = torch.concat([img.unsqueeze(0) for img, _ in anchor_subset], dim=0).to(device)

    # Define the models and datasets
    models = [resnet, mobilenet, squeezenet, ]
    datasets = [("train", cifar10_train), ("val", cifar10_val), ("test", cifar10_test)]
    model_names = ["resnet", "mobilenet", "squeezenet", ]

    # Loop through each model and dataset
    for model, model_name in zip(models, model_names):
        # Calculate the anchor matrix for each model
        anchors_embedded = model.features(anchor_subset_matrix) if model_name != "resnet" else model(anchor_subset_matrix)

        for dataset_type, dataset in datasets:
            # Construct the filename base for each combination (train/val/test)
            filename_base = f"data/own_reps_{model_name}/{dataset_type}/batch.pth"
            os.makedirs(os.path.dirname(filename_base), exist_ok=True)
            # Call the function to get relative representations and save them in batches
            get_relative_representations(dataset, anchors_embedded, model, filename_base, use_own_reps=True)
