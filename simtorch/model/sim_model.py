"""
Thin wrapper over `torch.nn.Module()` to add forwards hooks and store
activations as attributes
"""
from functools import partial
from typing import Dict, List, Union
from warnings import warn

import torch


class SimilarityModel:
    def __init__(
        self,
        model: torch.nn.Module,
        model_name: Union[None, str] = None,
        layers_to_include: Union[None, List[str]] = None,
        device: Union[torch.cuda.device, str] = "cpu",
    ):
        self.model = model
        self.device = device
        self.layers_to_include = [] if layers_to_include is None else layers_to_include
        if not self.layers_to_include:
            warn("No layers have been added for similarity metric computation.")

        self.model.to(self.device)
        self.model.eval()
        self.model_name: str = model_name if model_name is not None else self.model.__class__.__name__
        self.model_activations: Dict[str, torch.tensor] = {}

        self.hook_model()

    def hook_model(self):
        for name, layer in self.model.named_modules():
            if any([x in name for x in self.layers_to_include]):
                layer.register_forward_hook(partial(self._add_hook, name))

    def _add_hook(self, name, layer, input, output):
        self.model_activations[name] = output

    @property
    def n_layers(self):
        if len(self.model_activations) == 0:
            warn(
                message="Model activations are empty, there's a chance `model.forward()` hasn't been run yet."
            )
        return len(self.model_activations.keys())
