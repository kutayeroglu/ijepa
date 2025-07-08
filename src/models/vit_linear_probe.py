import torch
import torch.nn as nn


class LinearProbeModel(nn.Module):
    def __init__(self, encoder, embed_dim=1280, num_classes=1000):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            # Get patch embeddings: [batch, num_pathces, embed_dim]
            x = self.encoder(x)

        # I-JEPA uses average pooling (no CLS token)
        x = x.mean(dim=1)  # [batch, embed_dim]

        return self.classifier(x)
