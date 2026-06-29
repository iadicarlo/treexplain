"""Tests for deforestation model."""
import torch
import pytest
from treexplain.model.deforestation import DeforestationModel


class TestDeforestationModel:
    def test_model_forward(self):
        """Test model forward pass."""
        model = DeforestationModel(in_channels=6)
        x = torch.randn(1, 6, 256, 256)
        output = model(x)
        assert output.shape == (1, 1, 256, 256)
        assert (output >= 0).all()
        assert (output <= 1).all()

    def test_model_architecture(self):
        """Test model has expected architecture."""
        model = DeforestationModel()
        assert hasattr(model, 'enc1')
        assert hasattr(model, 'enc2')
        assert hasattr(model, 'bottleneck')
        assert hasattr(model, 'dec1')
