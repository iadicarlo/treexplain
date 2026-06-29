"""Tests for XAI module."""
import torch
import pytest
from treexplain.model.deforestation import DeforestationModel
from treexplain.xai.explainer import TreeXplainExplainer


class TestTreeXplainExplainer:
    @pytest.fixture
    def model_and_explainer(self):
        model = DeforestationModel(in_channels=6)
        return model, TreeXplainExplainer(model)

    def test_explainer_init(self, model_and_explainer):
        model, explainer = model_and_explainer
        assert explainer.model is model
        assert explainer.background is None

    def test_explain_gradcam(self, model_and_explainer):
        model, explainer = model_and_explainer
        input_tensor = torch.randn(1, 6, 256, 256)
        attribution = explainer.explain_gradcam(input_tensor)
        assert attribution.shape == (1, 6, 256, 256)

    def test_generate_report(self, model_and_explainer):
        model, explainer = model_and_explainer
        input_tensor = torch.randn(1, 6, 256, 256)
        report = explainer.generate_report(input_tensor, methods=["gradcam"])
        assert "gradcam" in report
