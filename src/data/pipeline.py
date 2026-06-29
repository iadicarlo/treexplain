"""Data pipeline for TreeXplain."""
import os
from pathlib import Path
from typing import Tuple, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from treexplain.config import DATA_DIR, INPUT_BANDS, IMAGE_SIZE
from treexplain.data.fetcher import fetch_sentinel2
from treexplain.data.preprocessor import SatellitePreprocessor


class DeforestationDataset(Dataset):
    """PyTorch Dataset for deforestation detection."""
    
    def __init__(
        self, 
        images: list, 
        labels: list, 
        transform: Optional[Callable] = None,
        augment: bool = False
    ):
        """Initialize dataset.
        
        Args:
            images: List of numpy arrays (channels, height, width)
            labels: List of labels (0 or 1)
            transform: Optional transform function
            augment: Whether to apply data augmentation
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
        # Simple augmentation: random flip
        self.augmentation_functions = [
            self._random_flip,
            self._random_rotation,
        ]
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx].copy()
        label = self.labels[idx]
        
        # Apply augmentation
        if self.augment:
            for aug_func in self.augmentation_functions:
                if np.random.rand() > 0.5:
                    image = aug_func(image)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float()
        
        # Apply transform if provided
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label
    
    def _random_flip(self, image: np.ndarray) -> np.ndarray:
        """Random horizontal and/or vertical flip."""
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1)  # Horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=2)  # Vertical flip
        return image
    
    def _random_rotation(self, image: np.ndarray) -> np.ndarray:
        """Random 90 degree rotation."""
        rotations = [0, 1, 2, 3]  # 0, 90, 180, 270 degrees
        k = np.random.choice(rotations)
        return np.rot90(image, k=k, axes=(1, 2))


class DataPipeline:
    """Complete data pipeline for TreeXplain."""
    
    def __init__(
        self, 
        bbox: list = None,
        days_back: int = 30,
        target_size: int = 256,
        batch_size: int = 8,
        test_size: float = 0.2,
        random_seed: int = 42
    ):
        """Initialize data pipeline.
        
        Args:
            bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
            days_back: Number of days to look back for data
            target_size: Target image size
            batch_size: Batch size for DataLoader
            test_size: Fraction of data for testing
            random_seed: Random seed for reproducibility
        """
        self.bbox = bbox or [-62.2159, -3.4653, -62.0, -3.2]  # Default: Amazon
        self.days_back = days_back
        self.target_size = target_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_seed = random_seed
        
        self.preprocessor = SatellitePreprocessor(
            target_size=target_size,
            bands=INPUT_BANDS
        )
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def fetch_and_preprocess(self, limit: int = 20) -> Tuple[list, list]:
        """Fetch STAC items and preprocess into images and labels.
        
        Args:
            limit: Maximum number of STAC items to fetch
            
        Returns:
            Tuple of (images, labels)
        """
        # Fetch STAC items
        stac_items = fetch_sentinel2(
            bbox=self.bbox,
            days_back=self.days_back,
            limit=limit
        )
        
        # For demo purposes, create synthetic labels
        # In production, you would have real labels from ground truth data
        def create_synthetic_label(item):
            # Simple heuristic: items with "cloud" in properties get label 0
            # This is just for demonstration - replace with real labeling
            if stac_items.index(item) % 3 == 0:
                return 1  # Deforestation
            return 0  # No deforestation
        
        # Create dataset
        images, labels = self.preprocessor.create_dataset_from_stac(
            stac_items,
            label_function=create_synthetic_label
        )
        
        return images, labels
    
    def create_datasets(self, images: list, labels: list) -> None:
        """Create train, validation, and test datasets.
        
        Args:
            images: List of images
            labels: List of labels
        """
        from sklearn.model_selection import train_test_split
        
        # Split into train+val and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            images, labels,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=labels
        )
        
        # Split train+val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.test_size / (1 - self.test_size),
            random_state=self.random_seed,
            stratify=y_train_val
        )
        
        # Create datasets
        self.train_dataset = DeforestationDataset(
            X_train, y_train,
            augment=True
        )
        self.val_dataset = DeforestationDataset(
            X_val, y_val,
            augment=False
        )
        self.test_dataset = DeforestationDataset(
            X_test, y_test,
            augment=False
        )
    
    def get_dataloaders(self) -> dict:
        """Get DataLoaders for all datasets.
        
        Returns:
            Dictionary with train, val, test DataLoaders
        """
        if None in [self.train_dataset, self.val_dataset, self.test_dataset]:
            raise ValueError("Datasets not created. Call create_datasets() first.")
        
        return {
            'train': DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2
            )
        }
    
    def get_sample_data(self, num_samples: int = 10) -> Tuple[list, list]:
        """Generate sample data for testing.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for i in range(num_samples):
            # Create random image with 6 channels
            image = np.random.rand(num_samples, 6, self.target_size, self.target_size)
            # Add some structure to make it more realistic
            for c in range(6):
                # Create some patterns
                x, y = np.meshgrid(np.linspace(0, 1, self.target_size), 
                                 np.linspace(0, 1, self.target_size))
                image[i, c] = 0.3 + 0.7 * (x * y + np.random.randn(self.target_size, self.target_size) * 0.1)
                image[i, c] = np.clip(image[i, c], 0, 1)
            
            # Label: 1 if "deforestation pattern" detected (random for demo)
            label = 1 if i % 3 == 0 else 0
            labels.append(label)
        
        return images, labels
