import os
import torch

from models.resnet import ResNet18
from models.preact_resnet import PreActResNet18


class Box:

    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.num_classes = opt.num_classes
        self.model_name = getattr(opt, "model", "resnet18")
        self._save_path = opt.save_dir
        os.makedirs(self._save_path, exist_ok=True)

    def get_save_path(self):
        return self._save_path

    def _load_raw(self):
        sd = torch.load(self.opt.ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for key in ("netC", "model", "net", "state_dict"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        return sd

    def _build_model(self):
        if self.model_name == "resnet18":
            return ResNet18(num_classes=self.num_classes)
        if self.model_name == "preactresnet18":
            return PreActResNet18(num_classes=self.num_classes)
        raise ValueError(f"Unsupported --model: {self.model_name}")

    def get_state_dict(self):
        classifier = self._build_model()
        classifier.load_state_dict(self._load_raw())
        classifier = classifier.to(self.device).eval()
        return None, None, classifier
