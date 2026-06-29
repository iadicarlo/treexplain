"""FastAPI backend for TreeXplain."""
import io
import base64
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from treexplain.config import MODELS_DIR
from treexplain.model.deforestation import DeforestationModel
from treexplain.xai.explainer import TreeXplainExplainer


app = FastAPI(
    title="TreeXplain API",
    description="API for explainable deforestation detection",
    version="0.1.0",
)

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model at startup (will need to train one first)
# For now, create a dummy model
model = DeforestationModel()
explainer = TreeXplainExplainer(model)


class PredictionRequest(BaseModel):
    """Request for prediction."""
    image: str  # Base64 encoded numpy array
    explain: bool = True
    methods: list[str] = ["shap", "gradcam"]


class PredictionResponse(BaseModel):
    """Response with prediction and explanation."""
    prediction: float
    probability: float
    explanation: dict[str, Any] | None = None
    message: str = "Success"


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on input image.
    
    Args:
        request: PredictionRequest with base64 encoded image
        
    Returns:
        PredictionResponse with results
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = np.load(io.BytesIO(image_data))
        
        # Convert to tensor
        if image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            output = model(image)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
        
        # Generate explanation if requested
        explanation = None
        if request.explain:
            # Set background for SHAP (simplified)
            if "shap" in request.methods:
                explainer.set_background(torch.zeros(5, *image.shape[1:]))
            
            report = explainer.generate_report(image, methods=request.methods)
            explanation = report
        
        return PredictionResponse(
            prediction=float(prediction),
            probability=probability,
            explanation=explanation,
            message="Prediction successful"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
