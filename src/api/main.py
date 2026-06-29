"""FastAPI backend for TreeXplain."""
import io
import base64
import time
from pathlib import Path
from typing import Any, List, Optional, Union
from datetime import datetime

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from treexplain.config import MODELS_DIR
from treexplain.model.deforestation import DeforestationModel
from treexplain.xai.explainer import TreeXplainExplainer
from treexplain.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfoResponse,
    STACSearchRequest,
    STACItemResponse,
    ErrorResponse,
    BoundingBox
)
from treexplain.data.fetcher import fetch_sentinel2


app = FastAPI(
    title="TreeXplain API",
    description="API for explainable deforestation detection",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model at startup
print("Loading model...")
model = DeforestationModel()
explainer = TreeXplainExplainer(model)

# Try to load trained model if available
trained_model_path = MODELS_DIR / "deforestation_model_final.pt"
if trained_model_path.exists():
    try:
        model.load_state_dict(torch.load(trained_model_path, map_location='cpu'))
        print(f"Loaded trained model from {trained_model_path}")
    except Exception as e:
        print(f"Could not load trained model: {e}")
else:
    print("No trained model found, using new model")

model.eval()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "name": "TreeXplain API",
        "version": "0.1.0",
        "description": "Explainable Deforestation Detection API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="0.1.0",
        timestamp=datetime.now().isoformat()
    )


@app.get("/model", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    num_parameters = sum(p.numel() for p in model.parameters())
    
    return ModelInfoResponse(
        name="DeforestationModel",
        version="0.1.0",
        architecture="U-Net",
        input_channels=6,
        input_size=256,
        parameters=num_parameters,
        trained_on="sample_data" if trained_model_path.exists() else None,
        accuracy=None  # Will be updated after training
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on input image.
    
    Args:
        request: PredictionRequest with base64 encoded image
        
    Returns:
        PredictionResponse with results
    """
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = np.load(io.BytesIO(image_data))
        
        # Convert to tensor
        if image.ndim == 3:
            # Check if HWC or CHW
            if image.shape[0] in [256, 512, 1024]:  # Likely HWC
                image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(image)
            probability = output.item()
            prediction = 1 if probability > request.threshold else 0
        
        # Generate explanation if requested
        explanation = None
        if request.explain:
            # Set background for SHAP (simplified)
            if "shap" in request.methods:
                explainer.set_background(torch.zeros(5, *image.shape[1:]))
            
            report = explainer.generate_report(image, methods=request.methods)
            explanation = report
        
        processing_time = time.time() - start_time
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=probability,
            explanation=explanation,
            model_version="0.1.0",
            processing_time=processing_time,
            message="Prediction successful"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions.
    
    Args:
        request: BatchPredictionRequest with list of images
        
    Returns:
        BatchPredictionResponse with results
    """
    start_time = time.time()
    
    try:
        predictions = []
        probabilities = []
        
        for image_b64 in request.images:
            # Decode base64 image
            image_data = base64.b64decode(image_b64)
            image = np.load(io.BytesIO(image_data))
            
            # Convert to tensor
            if image.ndim == 3:
                if image.shape[0] in [256, 512, 1024]:  # Likely HWC
                    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = model(image)
                probability = output.item()
                prediction = 1 if probability > request.threshold else 0
            
            predictions.append(int(prediction))
            probabilities.append(probability)
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            processing_time=processing_time,
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/stac/search", response_model=List[STACItemResponse])
async def search_stac(request: STACSearchRequest):
    """Search STAC catalog for satellite imagery.
    
    Args:
        request: STACSearchRequest with search parameters
        
    Returns:
        List of STAC items
    """
    try:
        # Convert bbox if needed
        if isinstance(request.bbox, list):
            bbox = request.bbox
        else:
            bbox = request.bbox.to_list()
        
        # Fetch from STAC
        items = fetch_sentinel2(
            bbox=bbox,
            days_back=30,  # Default to 30 days
            limit=request.limit
        )
        
        # Convert to response format
        responses = []
        for item in items:
            responses.append(STACItemResponse(
                id=item.id,
                collection=item.collection_id,
                datetime=str(item.datetime) if item.datetime else "",
                bbox=list(item.bbox) if item.bbox else [],
                properties=dict(item.properties) if item.properties else {},
                assets={k: {"href": v.href} for k, v in item.assets.items()} if item.assets else {}
            ))
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"STAC search failed: {str(e)}"
        )


@app.get("/stac/search", response_model=List[STACItemResponse])
async def search_stac_get(
    min_lon: float = Query(..., description="Minimum longitude"),
    min_lat: float = Query(..., description="Minimum latitude"),
    max_lon: float = Query(..., description="Maximum longitude"),
    max_lat: float = Query(..., description="Maximum latitude"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of items")
):
    """Search STAC catalog via GET request."""
    try:
        bbox = [min_lon, min_lat, max_lon, max_lat]
        items = fetch_sentinel2(
            bbox=bbox,
            days_back=30,
            limit=limit
        )
        
        responses = []
        for item in items:
            responses.append(STACItemResponse(
                id=item.id,
                collection=item.collection_id,
                datetime=str(item.datetime) if item.datetime else "",
                bbox=list(item.bbox) if item.bbox else [],
                properties=dict(item.properties) if item.properties else {},
                assets={k: {"href": v.href} for k, v in item.assets.items()} if item.assets else {}
            ))
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"STAC search failed: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            code=exc.status_code,
            details=None
        ).model_dump()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
