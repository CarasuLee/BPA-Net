import torch
from torch import nn

class FrozenBackboneModel(nn.Module):
    """Wrap a frozen image backbone with a trainable caption head."""

    def __init__(self, backbone: nn.Module, head: nn.Module, trainable: bool = False):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.trainable = trainable
        if not self.trainable:
            self.freeze_backbone()
        else:
            # Explicitly unfreeze if training is requested
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.train()

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()  # keep backbone frozen
        return self

    def forward(self, mode, images, **kwargs):
        if not self.trainable:
            with torch.no_grad():
                feats = self.backbone(images)
        else:
            feats = self.backbone(images)
        return self.head(mode=mode, images=feats, **kwargs)

    def init_state(self, b_s, device):
        return self.head.init_state(b_s, device)

    def step(self, *args, **kwargs):
        return self.head.step(*args, **kwargs)
