import torch.nn as nn

class LinearProbeModel(nn.Module):
    def __init__(self, encoder, embed_dim=1280, num_classes=1000):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Get encoder output
        x = self.encoder(x)
        
        # I-JEPA uses average pooling (no CLS token)
        # x shape: [batch_size, num_patches, embed_dim]
        x = x.mean(dim=1)  # Global average pooling
        
        # Linear classification
        x = self.classifier(x)
        return x