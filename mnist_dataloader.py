import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pandas as pd


def load_mnist_data(train_path: str, test_path: str):
    """
    Loads MNIST dataset from CSV files and returns processed datasets.
    
    Args:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the test dataset CSV file.
    
    Returns:
        Tuple: Processed training and test images and labels, along with mean and standard deviation of the training set.
    """
    df_train = pd.read_csv(train_path, header=None)
    df_test = pd.read_csv(test_path, header=None)

    # Extract labels (first column) and reshape images into (1, 28, 28) format
    y_train = df_train[0].to_numpy()
    X_train = df_train.drop(columns=0).values.reshape(-1, 1, 28, 28)
    y_test = df_test[0].to_numpy()
    X_test = df_test.drop(columns=0).values.reshape(-1, 1, 28, 28)
    
    # Normalize pixel values to range [-1, 1]
    X_train = ((X_train / 255.0) * 2) - 1
    X_test = ((X_test / 255.0) * 2) - 1
    
    return X_train, y_train, X_test, y_test


class MnistDataset(Dataset):
    """
    Custom dataset class for handling MNIST data.
    """
    def __init__(self, images, labels, transform=None):
        """
        Initializes the dataset.
        
        Args:
            images (numpy.ndarray): Array of image data.
            labels (numpy.ndarray): Array of corresponding labels.
            transform (callable, optional): Transform to be applied on images.
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        """
        Retrieves an image-label pair by index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            Tuple: Image and label at the specified index.
        """
        img = self.images[idx].reshape((28, 28)).astype(np.uint8) # Convert to uint8 format
        if self.transform:
            img = self.transform(img) # Apply transformation if provided
        label = self.labels[idx]
        return img, label


def get_dataloaders(train_path: str, test_path: str, batch_size=256):
    """
    Loads datasets and returns DataLoaders for training and testing.
    
    Args:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the test dataset CSV file.
        batch_size (int, optional): Batch size for DataLoader. Default is 256.
    
    Returns:
        Tuple[DataLoader, DataLoader]: DataLoaders for training and test datasets.
    """
    # Load MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data(train_path, test_path)

    # Define transformation pipeline for normalization
    train_transform = transforms.Compose(
                    [
                    transforms.ToTensor()
                    ])

    val_transform = transforms.Compose(
                    [
                    transforms.ToTensor(),
                    ])

    test_transform = val_transform # Use the same transformation for validation and test sets

    # Create training dataset and DataLoader
    dataset = MnistDataset(X_train, y_train, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Create test dataset and DataLoader
    test_dataset = MnistDataset(X_test, y_test, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return dataloader, test_dataloader