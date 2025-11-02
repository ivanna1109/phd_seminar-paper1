import torch
import torch.nn.functional as F

class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_coeff=5e-3, batch_size=None):
        super().__init__()
        # Lambda reguliše van-dijagonalne članove (smanjuje redundanciju)
        self.lambda_coeff = lambda_coeff
        
        # Ključno za normalizaciju batcheva
        if batch_size is None:
            print("Upozorenje: Nije definisana veličina batch-a za BarlowTwinsLoss.")
            
    def forward(self, z1, z2):
        """
        Računa Barlow Twins gubitak za dva skupa reprezentacija (z1 i z2).
        
        Args:
            z1 (torch.Tensor): Reprezentacije prvog pogleda (Batch_size, Feature_dim)
            z2 (torch.Tensor): Reprezentacije drugog pogleda (Batch_size, Feature_dim)
        """
        
        # 1. Normalizacija Batch-a (z-score normalizacija)
        # Srednja vrednost i std dev se računaju duž dimenzije Batch-a
        epsilon = 1e-5
        z1_norm = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + epsilon)
        z2_norm = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + epsilon)
        
        # 2. Računanje unakrsno-korelacijske matrice (C)
        # Matrica C je dimenzije (Feature_dim, Feature_dim)
        # Cij = Suma_k(z1_norm[k, i] * z2_norm[k, j]) / Batch_size
        batch_size = z1.size(0)
        c = torch.matmul(z1_norm.T, z2_norm) / batch_size
        
        # 3. Računanje dijagonalnih i van-dijagonalnih članova
        # Dijagonalni članovi (Težnja ka 1)
        # Sum_i (1 - C_ii)^2
        diag_loss = torch.sum((1. - torch.diag(c))**2)
        
        # Van-dijagonalni članovi (Težnja ka 0, sprečava kolaps)
        # Sum_i Sum_{j!=i} C_ij^2
        # Da bismo ovo izračunali, možemo uzeti sve članove na kvadrat i oduzeti dijagonalne
        off_diag_loss = torch.sum(c.triu(diagonal=1)**2) + torch.sum(c.tril(diagonal=-1)**2)
        
        # 4. Finalni Gubitak
        loss = diag_loss + self.lambda_coeff * off_diag_loss
        return loss