import torch.nn as nn


class MaskGenerator(nn.Module):

    def __init__(self, init_mask, classifier):
        super().__init__()
        self._EPSILON = 1e-7
        self.classifier = classifier
        self.mask_tanh = nn.Parameter(init_mask.clone().detach().requires_grad_(True))

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self._EPSILON) + 0.5
