import torch
import torch.nn as nn

class SSLModel(nn.Module):
    """
    Glavni Self-Supervised Learning Model.
    Kombinuje Feature Extractor (Encoder) i Projekcionu Glavu (Projection Head).
    """
    def __init__(self, encoder: nn.Module, projection_head: nn.Module):
        super(SSLModel, self).__init__()
        self.encoder = encoder
        self.projection_head = projection_head
    
    def forward(self, x):
        # 1. Propuštanje kroz enkoder (feature extractor)
        # S obzirom da get_ssl_encoder vraća model koji ima uklonjen FC sloj
        # i već uključuje Global Average Pooling, izlaz je Feature Vector.
        features = self.encoder(x) 
        
        # 2. Propuštanje Feature Vektora kroz projekcionu glavu
        projection = self.projection_head(features)
        
        return projection # Vektor projekcije