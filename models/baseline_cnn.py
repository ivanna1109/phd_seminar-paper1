import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    CNN arhitektura srednjeg kapaciteta (Mini-VGG inspirisana) sa Batch Normalization-om.
    Cilj: Rešavanje underfittinga
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()      
        
        self.features = nn.Sequential(
            # Blok 1: 224x224x3 -> 112x112x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # Dodajemo Batch Normalization
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Blok 2: 112x112x64 -> 56x56x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Dodajemo ponovljenu konvoluciju za veći kapacitet
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Blok 3: 56x56x128 -> 28x28x256
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Blok 4: 28x28x256 -> 14x14x512
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            # Global Average Pooling: (B, 512, 14, 14) -> (B, 512, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        # Ulazna dimenzija je sada 512 (zbog 512 kanala u poslednjem Conv sloju)
        input_dim_classifier = 512 
        hidden_dim_classifier = 1024
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim_classifier, hidden_dim_classifier), 
            nn.ReLU(),
            nn.Dropout(0.6), # Povećavamo Dropout za jaču regulaciju
            nn.Linear(hidden_dim_classifier, num_classes) 
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x