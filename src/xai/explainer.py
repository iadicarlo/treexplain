"""Explainable AI for deforestation model."""
import torch
import shap
import numpy as np
from captum.attr import GradientSHAP, IntegratedGradients, LayerGradCam


class TreeXplainExplainer:
    """Generate explanations for deforestation model predictions.
    
    Supports multiple XAI methods:
    - SHAP: Feature importance
    - GradCAM: Visual attention
    - Integrated Gradients: Pixel-level attribution
    """
    
    def __init__(self, model):
        """Initialize explainer.
        
        Args:
            model: Deforestation model
        """
        self.model = model
        self.background = None
    
    def set_background(self, background):
        """Set background data for SHAP.
        
        Args:
            background: Background samples (batch, channels, height, width)
        """
        self.background = background
    
    def explain_shap(self, input_tensor, n_samples=10):
        """Generate SHAP explanation.
        
        Args:
            input_tensor: Input to explain (1, channels, height, width)
            n_samples: Number of samples for approximation
            
        Returns:
            SHAP values
        """
        if self.background is None:
            raise ValueError("Background data not set. Call set_background() first.")
        
        explainer = shap.DeepExplainer(self.model, self.background)
        return explainer.shap_values(input_tensor)
    
    def explain_gradcam(self, input_tensor, target_layer=None):
        """Generate GradCAM explanation.
        
        Args:
            input_tensor: Input to explain (1, channels, height, width)
            target_layer: Layer to compute CAM for
            
        Returns:
            GradCAM attribution
        """
        if target_layer is None:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break
        
        gradcam = LayerGradCam(self.model, target_layer)
        return gradcam.attribute(input_tensor, target=0)
    
    def explain_integrated_gradients(self, input_tensor, baselines=None, n_steps=50):
        """Generate Integrated Gradients explanation.
        
        Args:
            input_tensor: Input to explain (1, channels, height, width)
            baselines: Baseline for integration (defaults to zeros)
            n_steps: Number of integration steps
            
        Returns:
            Attribution
        """
        if baselines is None:
            baselines = torch.zeros_like(input_tensor)
        
        ig = IntegratedGradients(self.model)
        return ig.attribute(input_tensor, baselines=baselines, target=0, n_steps=n_steps)
    
    def generate_report(self, input_tensor, methods=["shap", "gradcam"]):
        """Generate comprehensive explanation report.
        
        Args:
            input_tensor: Input to explain
            methods: List of methods to use
            
        Returns:
            Dictionary with all explanations
        """
        report = {}
        
        if "shap" in methods and self.background is not None:
            shap_values = self.explain_shap(input_tensor)
            report["shap"] = shap_values.cpu().numpy().tolist()
            
        if "gradcam" in methods:
            gradcam = self.explain_gradcam(input_tensor)
            report["gradcam"] = gradcam.cpu().numpy().tolist()
            
        if "integrated_gradients" in methods:
            ig = self.explain_integrated_gradients(input_tensor)
            report["integrated_gradients"] = ig.cpu().numpy().tolist()
        
        return report
