"""Data preprocessing for satellite imagery."""
import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import rioxarray
import xarray as xr


class SatellitePreprocessor:
    """Preprocess satellite imagery for model input."""
    
    def __init__(self, target_size: int = 256, bands: list = None):
        """Initialize preprocessor.
        
        Args:
            target_size: Target size for patches (default: 256)
            bands: List of bands to extract (default: Sentinel-2 bands)
        """
        self.target_size = target_size
        self.bands = bands or ["B02", "B03", "B04", "B08", "B11", "B12"]
    
    def normalize(self, array: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range.
        
        Args:
            array: Input array
            
        Returns:
            Normalized array
        """
        if array.dtype == np.uint16:
            array = array.astype(np.float32) / 65535.0
        elif array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            # Assume already normalized or float
            array = array.astype(np.float32)
            if array.max() > 1.0:
                array = array / array.max()
        return array
    
    def extract_patches(self, image_path: Path, patch_size: int = 256, stride: int = 128) -> list:
        """Extract patches from a large satellite image.
        
        Args:
            image_path: Path to the satellite image (GeoTIFF)
            patch_size: Size of each patch
            stride: Stride between patches
            
        Returns:
            List of patches (numpy arrays)
        """
        with rasterio.open(image_path) as src:
            patches = []
            
            # Calculate number of patches in each dimension
            height = src.height
            width = src.width
            
            for i in range(0, height - patch_size + 1, stride):
                for j in range(0, width - patch_size + 1, stride):
                    window = Window(j, i, patch_size, patch_size)
                    patch = src.read(window=window)
                    
                    # Select only the bands we need
                    if len(self.bands) < src.count:
                        # Map band names to indices
                        band_indices = []
                        for band in self.bands:
                            try:
                                idx = src.descriptions.index(band) if src.descriptions else None
                                if idx is None:
                                    # Try to find by name in metadata
                                    for k, v in src.tags().items():
                                        if band in str(v):
                                            idx = int(k.split('_')[-1]) - 1
                                            break
                                if idx is None:
                                    idx = self.bands.index(band)  # Fallback
                                band_indices.append(idx)
                            except (ValueError, AttributeError):
                                band_indices.append(self.bands.index(band))
                        patch = patch[band_indices]
                    
                    # Normalize
                    patch = self.normalize(patch)
                    patches.append(patch)
            
            return patches
    
    def create_dataset_from_stac(self, stac_items: list, label_function=None) -> Tuple[list, list]:
        """Create dataset from STAC items.
        
        Args:
            stac_items: List of STAC items
            label_function: Function to generate labels from STAC items
            
        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []
        
        for item in stac_items:
            try:
                # Open the item with rioxarray
                xds = xr.open_dataset(item, engine="pystac")
                
                # Select bands
                band_data = []
                for band in self.bands:
                    if band in xds.data_vars:
                        band_data.append(xds[band].values)
                
                if band_data:
                    image = np.stack(band_data, axis=0)
                    image = self.normalize(image)
                    
                    # Resize to target size if needed
                    if image.shape[1:] != (self.target_size, self.target_size):
                        image = self.resize_image(image, self.target_size)
                    
                    images.append(image)
                    
                    # Generate label if function provided
                    if label_function:
                        label = label_function(item)
                        labels.append(label)
                    else:
                        labels.append(0)  # Default: no deforestation
                        
            except Exception as e:
                print(f"Error processing STAC item {item.id}: {e}")
                continue
        
        return images, labels
    
    def resize_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """Resize image to target size using interpolation.
        
        Args:
            image: Input image (channels, height, width)
            target_size: Target size
            
        Returns:
            Resized image
        """
        from skimage.transform import resize
        
        resized = []
        for channel in image:
            resized_channel = resize(
                channel, 
                (target_size, target_size), 
                mode='reflect',
                anti_aliasing=True
            )
            resized.append(resized_channel)
        
        return np.stack(resized, axis=0)
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.
        
        Args:
            image: Input image (channels, height, width)
            
        Returns:
            PyTorch tensor (channels, height, width)
        """
        tensor = torch.from_numpy(image).float()
        return tensor
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess a batch of images.
        
        Args:
            images: List of numpy arrays
            
        Returns:
            Batch tensor (batch, channels, height, width)
        """
        tensors = [self.to_tensor(img) for img in images]
        return torch.stack(tensors, dim=0)
