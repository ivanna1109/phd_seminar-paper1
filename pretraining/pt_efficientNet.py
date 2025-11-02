"""
SECTION: Data Loading and Setup for Self-Supervised Pre-training

PURPOSE: This block initializes the necessary components (Transforms, Dataset, 
and DataLoader) to prepare the embryo image data for the Self-Supervised Learning (SSL) 
pre-training phase. It bridges the gap between the saved Parquet files and 
the PyTorch training loop.

DESCRIPTION:
1. Path Definition: Specifies the location of the pre-processed Parquet file 
   containing the training images ('train_images.parquet').
2. Transforms Initialization: The dual-view augmentation function (get_ssl_transforms) 
   is called, setting the standard input size (e.g., 224x224).
3. Dataset Creation: An instance of the custom EmbryoSSLDataset is created, 
   which is responsible for reading and decoding the images from the Parquet file 
   and applying the dual-view transforms.
4. DataLoader Configuration: The DataLoader is set up to handle batching (BATCH_SIZE), 
   shuffling (mandatory for training), and efficient parallel loading (NUM_WORKERS) 
   to feed the GPU/CPU efficiently. The output is 'ssl_train_dataloader', which 
   yields batches containing pairs of augmented image views (view1, view2).
"""
import sys

sys.path.append('/home/ivanam/Seminar1')
from data_preparation.ssl_transorms import get_ssl_transforms
from data_preparation.dataloaders import EmbryoSSLDataset, EmbryoSSLDataset2
from models.baseline_cnn import SimpleCNN
from models.pretrained_models import get_pretrained_model
from projection_head import ProjectionHeadBarlow as ProjectionHead
from ssl_encoders import get_ssl_encoder, get_encoder_output_dim
from ssl_model import SSLModel
from pretraining.barlow_loss import BarlowTwinsLoss
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

train_parquet_path = 'localWork/prepared_data/train_images.parquet' 

# Inicijalizacija transformacija
ssl_transforms_fn = get_ssl_transforms(input_size=224) 

# Kreiranje Dataset instance
ssl_train_dataset = EmbryoSSLDataset2(
    parquet_path=train_parquet_path,
    transforms=ssl_transforms_fn,
    minority_repeat_factor=5
)

# Kreiranje DataLoader-a (za efikasno dohvatanje podataka tokom treninga)
BATCH_SIZE = 128 # Počnite sa 64
NUM_WORKERS = 4 # Koristite 4 ili više ako je CPU jak za brže učitavanje

ssl_train_dataloader = DataLoader(
    ssl_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Obavezno za trening
    num_workers=NUM_WORKERS,
    pin_memory=True # Poboljšava performanse sa GPU
)

print(f"\nDataLoader spreman. Broj batch-eva po epohi: {len(ssl_train_dataloader)}")
print("Sledeći korak: Definisanje EmbryoNet/ResNet modela i pokretanje faze predučenja (pre-training)!")

print("\n--- Inicijalizacija baseline modela za test ---")

# 1. Baseline CNN
baseline_cnn = SimpleCNN(num_classes=2)
print(f"1. Simple CNN: Broj parametara: {sum(p.numel() for p in baseline_cnn.parameters()):,}")

# 2. ResNet (ne-predučeni, za SSL fazu)
resnet_model = get_pretrained_model('resnet18', num_classes=2, pre_trained=False)
print(f"2. ResNet-18: Broj parametara: {sum(p.numel() for p in resnet_model.parameters()):,}")

# 3. MobileNet (ne-predučeni, za SSL fazu)
mobilenet_model = get_pretrained_model('mobilenet_v3_large', num_classes=2, pre_trained=False)
print(f"3. MobileNetV3: Broj parametara: {sum(p.numel() for p in mobilenet_model.parameters()):,}")

# 4. EfficientNet (ne-predučeni, za SSL fazu)
efficientnet_model = get_pretrained_model('efficientnet_b0', num_classes=2, pre_trained=False)
print(f"4. EfficientNet-B0: Broj parametara: {sum(p.numel() for p in efficientnet_model.parameters()):,}")

print("\nModeli su uspešno definisani i spremni za upotrebu u trening petlji.")

# --- Hiperparametri ---
MODEL_NAME = 'efficientnet_b0' # sad MobileNetV3Large
NUM_EPOCHS = 100          # varirati po potrebi
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 5e-4
SAVE_MODEL_PATH = f'pretraining/ssl_checkpoints/{MODEL_NAME}_embryo_ssl.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nKoristi se uređaj: {device}")

# 1. Kreiranje SSL Enkodera
try:
    encoder = get_ssl_encoder(MODEL_NAME, pre_trained=False)
    encoder_output_dim = get_encoder_output_dim(MODEL_NAME)
except NotImplementedError as e:
    print(f"Greška: {e}. Preskačem SSL trening za ovaj model.")
    # Možete ovde prekinuti ili dodati implementacije
    sys.exit() 


# 2. Kreiranje SSL Modela (Encoder + Projection Head)
projection_head = ProjectionHead(input_dim=encoder_output_dim, hidden_dim=2048, output_dim=2048) #uobicajeno za ovaj barlow
ssl_model = SSLModel(encoder, projection_head).to(device)

# 3. Postavljanje Optimizatora, Loss funkcije i Scheduler-a
optimizer = Adam(ssl_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = BarlowTwinsLoss(lambda_coeff=5e-3)

global_step = 0
initial_lr = 1e-6 # Vrlo mala početna vrednost LR

# Postavljanje scheduler-a da traje ukupno koraka (NUM_EPOCHS * len(dataloader))
total_steps = NUM_EPOCHS * len(ssl_train_dataloader)
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
WARMUP_EPOCHS = 5 # Broj epoha za zagrevanje
WARMUP_STEPS = WARMUP_EPOCHS * len(ssl_train_dataloader) # Ukupan broj batch-eva za Warmup
# --- Hiperparametri za Early Stopping ---
PATIENCE = 50         # Broj epoha koje se čekaju bez poboljšanja
MIN_DELTA = 0.001       # Minimalno poboljšanje gubitka (opadanje) da bi se resetovao brojač

# --- Inicijalizacija za Praćenje ---
best_loss = float('inf')
patience_counter = 0

# 4. Glavna Trening Petlja
print("\n--- Pokretanje Self-Supervised Predučenja (Barlow Twins) ---")
os.makedirs('pretraining/ssl_checkpoints', exist_ok=True) 

for epoch in range(1, NUM_EPOCHS + 1):
    ssl_model.train()
    total_loss = 0
    
    for step, (view1, view2) in enumerate(ssl_train_dataloader):
        
        # A) WARMUP FAZA
        if global_step < WARMUP_STEPS:
            # Ručno postavljamo Learning Rate (LR)
            # Linearno povećanje LR od initial_lr do LEARNING_RATE
            current_lr = initial_lr + (LEARNING_RATE - initial_lr) * (global_step / WARMUP_STEPS)
            
            # Postavljanje novog LR u optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        
        # B) COSINE ANNEALING FAZA
        else:
            # Kada je Warmup završen, Cosine Annealing preuzima kontrolu.
            # Vaš scheduler je definisan da se zove po batch-u (koraku).
            scheduler.step()

        view1 = view1.to(device)
        view2 = view2.to(device)
        
        optimizer.zero_grad()
        z_i = ssl_model(view1)
        z_j = ssl_model(view2)
        
        loss = criterion(z_i, z_j)
        
        loss.backward()
        optimizer.step()
        
        # IZMENA 2: Inkremetiranje globalnog brojača KORAKA (batch-eva)
        global_step += 1
        
        total_loss += loss.item()

    # ---------------------------------------------------------------------
    # Ispis i Early Stopping logika na kraju epohe
    # ---------------------------------------------------------------------
    
    avg_loss = total_loss / len(ssl_train_dataloader)
    current_lr_final = optimizer.param_groups[0]['lr'] # Ispisuje LR postavljen na zadnjem batch-u
    
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Avg Loss: {avg_loss:.4f} | LR: {current_lr_final:.6f}")

    # ---------------------------------------------------------------------
    #  EARLY STOPPING
    # ---------------------------------------------------------------------
    if avg_loss < best_loss - MIN_DELTA:
        # Poboljšanje: Sačuvaj model i resetuj brojač
        print(f"   [STOP] Loss poboljšan sa {best_loss:.4f} na {avg_loss:.4f}. Resetujem pult.")
        best_loss = avg_loss
        patience_counter = 0
        
        # Opciono: Sačuvajte najbolji model do sada
        torch.save(ssl_model.encoder.state_dict(), SAVE_MODEL_PATH.replace(".pth", f"bs{BATCH_SIZE}_best.pth"))
    
    else:
        # Nema poboljšanja: Povećaj brojač
        patience_counter += 1
        print(f" [STOP] Loss nije poboljšan. Pult: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n*** RANI PREKID: Gubitak se nije poboljšao u poslednjih {PATIENCE} epoha. ***")
            break # Prekida se glavna petlja

# 5. Čuvanje FINALNOG stanja predučenog enkodera (ako se trening nije prekinuo)
# Opciono možete sačuvati i model sa najboljim loss-om (SAVE_MODEL_PATH_best.pth)
if epoch < NUM_EPOCHS:
    print("Koristim model sa najboljim gubitkom (prekidanja) za finetuniranje.")
else:
    torch.save(ssl_model.encoder.state_dict(), SAVE_MODEL_PATH)
    print(f"\nPredučeni enkoder sačuvan u: {SAVE_MODEL_PATH}")