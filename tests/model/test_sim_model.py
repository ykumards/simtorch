import torch

from simtorch.model.sim_model import SimilarityModel


class DummyModel(torch.nn.Module):
    def __init__(self):
        self.layer1 = torch.nn.Linear(2, 4)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.layer2(self.act1(self.layer1(x)))


class TestSimModel:
    def test_init():
        dummy_model = DummyModel()

        sim_model = SimilarityModel(
            model=dummy_model,
            model_name="test_model",
            layers_to_include=["linear"],
            device="cpu"
        )

        assert isinstance(sim_model, SimilarityModel)
