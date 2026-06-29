"""Pydantic schemas for TreeXplain API."""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
import numpy as np


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    min_lon: float = Field(..., description="Minimum longitude")
    min_lat: float = Field(..., description="Minimum latitude")
    max_lon: float = Field(..., description="Maximum longitude")
    max_lat: float = Field(..., description="Maximum latitude")
    
    @classmethod
    def from_list(cls, bbox: List[float]) -> "BoundingBox":
        """Create BoundingBox from list [min_lon, min_lat, max_lon, max_lat]."""
        return cls(
            min_lon=bbox[0],
            min_lat=bbox[1],
            max_lon=bbox[2],
            max_lat=bbox[3]
        )
    
    def to_list(self) -> List[float]:
        """Convert to list format."""
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


class STACSearchRequest(BaseModel):
    """Request for STAC search."""
    bbox: Union[BoundingBox, List[float]] = Field(
        ...,
        description="Bounding box as list [min_lon, min_lat, max_lon, max_lat] or BoundingBox object"
    )
    start_date: Optional[str] = Field(
        None,
        description="Start date in YYYY-MM-DD format"
    )
    end_date: Optional[str] = Field(
        None,
        description="End date in YYYY-MM-DD format"
    )
    collections: List[str] = Field(
        ["sentinel-2-l2a"],
        description="STAC collections to search"
    )
    limit: int = Field(
        10,
        ge=1,
        le=100,
        description="Maximum number of items to return"
    )
    cloud_cover: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Maximum cloud cover percentage"
    )


class STACItemResponse(BaseModel):
    """Simplified STAC item response."""
    id: str
    collection: str
    datetime: str
    bbox: List[float]
    properties: Dict[str, Any]
    assets: Dict[str, Dict[str, str]]


class PredictionRequest(BaseModel):
    """Request for prediction."""
    image: str = Field(
        ...,
        description="Base64 encoded numpy array (CHW format)"
    )
    explain: bool = Field(
        True,
        description="Whether to generate explanations"
    )
    methods: List[str] = Field(
        ["shap", "gradcam"],
        description="List of XAI methods to use"
    )
    threshold: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Threshold for binary prediction"
    )


class ExplanationResponse(BaseModel):
    """Explanation data."""
    shap: Optional[List[List[List[List[float]]]]] = Field(
        None,
        description="SHAP values for each input feature"
    )
    gradcam: Optional[List[List[List[List[float]]]]] = Field(
        None,
        description="GradCAM attribution maps"
    )
    integrated_gradients: Optional[List[List[List[List[float]]]]] = Field(
        None,
        description="Integrated Gradients attribution"
    )


class PredictionResponse(BaseModel):
    """Response with prediction and explanation."""
    prediction: int = Field(
        ...,
        description="Binary prediction (0: no deforestation, 1: deforestation)"
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of deforestation"
    )
    explanation: Optional[ExplanationResponse] = Field(
        None,
        description="Explanation data if requested"
    )
    model_version: str = Field(
        "0.1.0",
        description="Model version"
    )
    processing_time: float = Field(
        ...,
        description="Processing time in seconds"
    )
    message: str = Field(
        "Success",
        description="Status message"
    )


class BatchPredictionRequest(BaseModel):
    """Request for batch prediction."""
    images: List[str] = Field(
        ...,
        description="List of base64 encoded numpy arrays"
    )
    explain: bool = Field(
        False,
        description="Whether to generate explanations (not recommended for large batches)"
    )
    threshold: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Threshold for binary prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[int] = Field(
        ...,
        description="List of binary predictions"
    )
    probabilities: List[float] = Field(
        ...,
        description="List of probabilities"
    )
    processing_time: float = Field(
        ...,
        description="Total processing time in seconds"
    )
    count: int = Field(
        ...,
        description="Number of predictions"
    )


class ModelInfoResponse(BaseModel):
    """Information about the model."""
    name: str = Field(
        "DeforestationModel",
        description="Model name"
    )
    version: str = Field(
        "0.1.0",
        description="Model version"
    )
    architecture: str = Field(
        "U-Net",
        description="Model architecture"
    )
    input_channels: int = Field(
        6,
        description="Number of input channels"
    )
    input_size: int = Field(
        256,
        description="Input image size"
    )
    parameters: int = Field(
        ...,
        description="Number of trainable parameters"
    )
    trained_on: Optional[str] = Field(
        None,
        description="Dataset used for training"
    )
    accuracy: Optional[float] = Field(
        None,
        description="Model accuracy on test set"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        "healthy",
        description="Service status"
    )
    model_loaded: bool = Field(
        True,
        description="Whether model is loaded"
    )
    version: str = Field(
        "0.1.0",
        description="API version"
    )
    timestamp: str = Field(
        ...,
        description="Current timestamp"
    )


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(
        ...,
        description="Error message"
    )
    code: int = Field(
        ...,
        description="Error code"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
