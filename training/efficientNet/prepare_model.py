import torch
from torchvision.models import efficientnet_b0
import torch.nn as nn
def clean_state_dict_keys(state_dict):
    """
    Preimenovanje ključeva iz formata SSL modela (npr. '0.X.Y...') 
    u format koji koristi torchvision EfficientNet (npr. 'features.X.Y...')
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        # Uklanjanje prefiksa (npr. '0.') i preimenovanje:
        if k.startswith('0.'):
            # Većina težina je u telu modela (features)
            new_k = k.replace('0.', 'features.', 1)
        elif k.startswith('1.'):
            # Poslednji konvolucioni deo (features.8) i glava/klasifikator
            # Standardni torchvision model ima: features.8.weight, features.8.1.weight, features.8.1.bias
            # Ako je ključ '1.0.weight', to bi mogao biti features.8.weight
            if 'classifier' in k:
                 # Zanemarite ključeve klasifikatora ako želite da ga zamenite kasnije.
                 # Ali ako su sačuvane samo 'features' težine (što je čest slučaj kod SSL),
                 # onda se moramo fokusirati samo na 'features' ključeve.
                 continue
            else:
                 # Ako vaš model ima samo 'features' i na kraju 'classifier', a svi features 
                 # ključevi počinju sa '0.', onda bi sve trebalo da bude u bloku '0.'.
                 # Na osnovu vaših grešaka, čini se da vaš model ima features (0.X) i pool/head (1.X).
                 # Međutim, najsigurnije je fokusirati se na 'features' koje ste obučavali.
                 
                 # Pošto vaš SSL checkpoint ima ključeve koji počinju sa '0.' za celo telo, 
                 # i oni se ne poklapaju sa torchvision, preimenovaćemo ih da odgovaraju:
                 
                 # Ključevi koje vaš model ima: '0.0.0.weight', '0.0.1.weight', ...
                 # Ključevi koje torchvision očekuje: 'features.0.0.weight', 'features.0.1.weight', ...

                 # Ponovo, na osnovu greške, svi "Unexpected keys" počinju sa '0.' ili '1.'
                 # Fokusirajmo se na najčešći slučaj u EfficientNetu (features)

                 # Ključevi od '0.0.0.weight' do '0.7.0.block...' postaju 'features.0.0.weight' do 'features.7.0.block...'
                 # Ključevi '0.8.0.weight' do '0.8.1.running_var' postaju 'features.8.0.weight' do 'features.8.1.running_var'
                 
                 new_k = k
                 # Ključevi '0.8.0.weight', '0.8.1.weight' se mapiraju na 'features.8.0.weight', 'features.8.1.weight'
                 # Telo modela u vašem SSL fajlu izgleda da se nalazi pod ključem '0'
                 
                 # Pokušajmo najjednostavnije preimenovanje:
                 # Zamena prvog broja sa 'features'
                 parts = k.split('.', 1)
                 if len(parts) > 1 and parts[0].isdigit():
                     new_k = 'features.' + parts[1]
                 else:
                     new_k = k
                 
        else:
            new_k = k
            
        # Dodatno, uklonite 'num_batches_tracked' jer to može izazvati 'Unexpected key' grešku
        if 'num_batches_tracked' in new_k:
            continue

        new_state_dict[new_k] = v
        
    return new_state_dict