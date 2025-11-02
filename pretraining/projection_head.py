import torch.nn as nn

class ProjectionHeadBarlow(nn.Module):
    """
    Projection Head implementacija prilagođena specifikacijama Barlow Twins.
    Koristi 3 linearna sloja, bez bias-a i sa affine=False na poslednjem BatchNorm sloju.
    
    Preporučene dimenzije: input_dim (npr. 1280), hidden_dim=2048, output_dim=2048
    """
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 2048):
        super(ProjectionHeadBarlow, self).__init__()
        
        # Sloj 1: Input -> Hidden
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False), # Bias=False preporučeno
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Sloj 2: Hidden -> Hidden
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False), # Bias=False preporučeno
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Sloj 3: Hidden -> Output
        # KLJUČNO: Poslednji BatchNorm NE SME imati učenje parametara (affine=False)
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False), # Bias=False preporučeno
            nn.BatchNorm1d(output_dim, affine=False)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x