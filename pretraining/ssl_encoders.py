import torch
import torch.nn as nn
from torchvision import models

def get_ssl_encoder(model_name: str, pre_trained: bool = False) -> nn.Module:
    """
    Kreira PyTorch model (Enkoder) za Self-Supervised Learning (SSL).
    Uklanja poslednji klasifikacioni sloj, vraćajući samo telo modela (feature extractor).
    """
    if model_name == 'resnet18':
        # ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None)
        
        encoder_layers = list(model.children())[:-1] 
        
        # 2. Dodavanje Flatten sloja NAKON Average Poolinga
        # Flatten preoblikuje iz (B, 512, 1, 1) u (B, 512)
        encoder_layers.append(nn.Flatten(start_dim=1)) 
        
        # 3. Spajanje slojeva u novu Sequential sekvencu
        model = nn.Sequential(*encoder_layers)
        
    elif model_name == 'mobilenet_v3_large':
        # MobileNetV3 (Large)
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pre_trained else None)
        print("Pribavljamo model mobileNet_v3_large...")
        model.classifier[3] = nn.Identity()
    
    # KORAK 2: Dodajemo Flatten() na kraj modela.
    # Pošto je ceo model sada Sequential (features + classifier), 
    # kreiramo novi Sequential model da bi se Flatten primenio nakon svega.

    # Sastavljamo ceo novi enkoder:
        encoder = nn.Sequential(
        model, # Ceo model, ali sa Identity umesto poslednjeg sloja
        nn.Flatten(start_dim=1) # Osiguravamo izlaz (B, 1280)
        )
    
        model = encoder
        
    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pre_trained else None)
        
        # EfficientNet telo je u 'features' sekciji. 'avgpool' i 'classifier' su odvojeni.
        # Spajamo features i global average pooling, uklanjajući classification head.
        
        # Novo telo modela (Encoder)
        model = nn.Sequential(
            model.features, # Konvolucioni slojevi
            model.avgpool,  # Global Average Pooling
            nn.Flatten(start_dim=1) # Flatten (umesto Dropout i Linear slojeva)
        )
        # SADA: Izlaz je feature vector dimenzije 1280 (nakon Global Average Poolinga)
        
    else:
        raise ValueError(f"Model {model_name} nije podržan za SSL enkoder.")
        
    # Svi ovi modeli sada vraćaju Feature Vector (enkoding slike), bez klasifikacije.
    return model

# --- POMOĆNA FUNKCIJA ZA ODREĐIVANJE IZLAZNE DIMENZIJE ---

def get_encoder_output_dim(model_name: str) -> int:
    """
    Vraća dimenziju feature vektora koji SSL enkoder generiše.
    Ovo je neophodno za inicijalizaciju ProjectionHead.
    """
    if model_name == 'resnet18':
        return 512
    elif model_name == 'mobilenet_v3_large':
        # Poslednji konvolucioni sloj (Squeeze-and-Excitation) daje 960 kanala,
        # ali klasifikator kreće od 1280 (MobileNetV3 specifičnost). Ostaćemo na 1280.
        return 1280 
    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0 izlaz nakon avgpool-a
        return 1280 
    elif model_name == 'simplecnn':
        # Iz SimpleCNN primera (64 * 28 * 28)
        return 50176
    else:
        raise ValueError(f"Model {model_name} nije podržan.")

# Sada možete kombinovati:
# encoder = get_ssl_encoder('resnet18', pre_trained=False)
# input_dim = get_encoder_output_dim('resnet18')
# projection_head = ProjectionHead(input_dim=input_dim)