import torch
import torch.nn as nn
from torchvision import models

def get_pretrained_model(model_name: str, num_classes: int = 2, pre_trained: bool = False):
    """
    Kreira PyTorch model koristeći torchvision.models, uklanjajući završni 
    klasifikacioni sloj za SSL pre-trening, ili ga zadržavajući/menjajući za 
    klasifikaciju/finotreniranje.
    """
    if model_name == 'resnet18':
        # ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None)
        # Menjanje poslednjeg linearnog sloja za 2 klase
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'mobilenet_v3_large':
        # MobileNetV3 (Large)
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pre_trained else None)
        # Menjanje poslednjeg linearnog sloja za 2 klase
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0 (Najmanja verzija)
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pre_trained else None)
        # Menjanje poslednjeg linearnog sloja za 2 klase
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} nije podržan.")
        
    return model