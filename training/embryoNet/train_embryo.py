import sys
sys.path.append('/home/ivanam/Seminar1')
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
import os
import pandas as pd
# 👇 KLJUČNA IZMENA 1: Uvoz EmbryoNet2 modela (Pretpostavka: u modelu)
from model import EmbryoNet2 
from torch.optim.lr_scheduler import CosineAnnealingLR
from prepare_dataset import prepare_supervised_dataloaders # Očekuje se da vraća 3 tenzora

# --- KONFIGURACIJA ---
MODEL_NAME = 'EmbryoNet2_TrainFromScratch' 
NUM_CLASSES = 2
NUM_EPOCHS = 75           # Povećano zbog niskog LR-a
BATCH_SIZE = 32           # Manji Batch Size za stabilnost treninga od nule
# 👇 KORIGOVANI HIPERPARAMETRI ZA TRENING OD NULE
LEARNING_RATE = 1e-4      # Drastično smanjeno za stabilnost
PATIENCE = 10             # Dodajemo Rano zaustavljanje
# -----------------------------------------------------------------------
TRAIN_PARQUET = 'localWork/prepared_data/train_images.parquet' 
VAL_SPLIT_RATIO = 0.2
SAVE_PATH = f'/home/ivanam/Seminar1/training/finetuned_models/embryo/{MODEL_NAME}_best.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nKoristi se uređaj: {device}")
# 👇 KLJUČNA IZMENA 2: Isključivanje oversamplinga (Oslanjamo se samo na Weighted Loss)
NUM_REPEATS_TRAIN = 1 

try:
    # Očekujemo da prepare_supervised_dataloaders VRAĆA SLIKE, DANE i LABELE
    train_loader, val_loader, class_weights = prepare_supervised_dataloaders(
        parquet_path=TRAIN_PARQUET,
        num_classes=NUM_CLASSES,
        val_split_ratio=VAL_SPLIT_RATIO,
        batch_size=BATCH_SIZE,
        minority_repeat_train=NUM_REPEATS_TRAIN
    )
except Exception as e:
    print(f"Greška pri pripremi podataka: {e}")
    sys.exit()

# --- 2. INICIJALIZACIJA MODELA ---
# Model se inicijalizuje sa nasumičnim težinama jer ne učitavamo SSL
model = EmbryoNet2(num_classes=NUM_CLASSES, final_softmax=False).to(device)

manual_weights = torch.tensor([1.0, 3.0], dtype=torch.float)
# --- 3. OPTIMIZATOR I GUBITAK ---
# Ostavljamo Weighted Loss za balansiranje, jer oversampling ne koristimo.
criterion = nn.CrossEntropyLoss(weight=manual_weights.to(device)) 
print("Koristi se CrossEntropyLoss sa izračunatim težinama klase!")

# Koristimo Adam kao u SimpleCNN kodu, ali sa novim, nižim LR-om
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# --- 4. TRENING PETLJA ---
print(f"\n--- Pokretanje Supervised Treninga za EmbryoNet2 model {MODEL_NAME} ---")
best_val_f1 = 0.0
training_history = []
patience_counter = 0
LOG_FILE_PATH = f'training/embryoNet/results/{MODEL_NAME}_training_log.csv'

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0
    train_preds = []
    train_labels = []

    # 👇 KLJUČNA IZMENA 3: Učitavanje tri tenzora iz Dataloadera
    for inputs, t_timestamps, labels in train_loader:
        inputs, t_timestamps, labels = inputs.to(device), t_timestamps.to(device), labels.to(device)
        
        optimizer.zero_grad()
        # 👇 KLJUČNA IZMENA 4: Pozivanje modela sa dva ulaza (slike, dan)
        outputs = model(inputs, t_timestamps) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
    avg_train_loss = total_loss / len(train_loader)
    train_f1 = f1_score(train_labels, train_preds, average='binary')
    train_accuracy = accuracy_score(train_labels, train_preds)
    
    # VALIDACIONA FAZA
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # 👇 KLJUČNA IZMENA 5: Učitavanje tri tenzora i u validaciji
        for inputs, t_timestamps, labels in val_loader:
            inputs, t_timestamps, labels = inputs.to(device), t_timestamps.to(device), labels.to(device)
            # 👇 KLJUČNA IZMENA 6: Pozivanje modela sa dva ulaza
            outputs = model(inputs, t_timestamps) 
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    val_accuracy = accuracy_score(all_labels, all_preds)
    
    epoch_results = {
        'epoch': epoch,
        'train_loss': avg_train_loss,
        'train_accuracy': train_accuracy,
        'train_f1': train_f1,
        'val_accuracy': val_accuracy,
        'val_f1': val_f1
    }
    training_history.append(epoch_results)

    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Loss: {avg_train_loss:.4f} | T_Acc: {train_accuracy:.4f} | V_Acc: {val_accuracy:.4f} | V_F1: {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        print("   --> Sačuvan najbolji model EmbryoNet2!")
    else:
        patience_counter += 1
        print(f"   --> Nema poboljšanja. Patience brojač: {patience_counter}/{PATIENCE}")

    scheduler.step()
    
    # Rano zaustavljanje
    if patience_counter >= PATIENCE:
        print(f"\n! RANO ZAUSTAVLJANJE ! Validacioni F1 se nije poboljšao {PATIENCE} epoha.")
        break

print("\n--- Supervised Trening završen. ---")
try:
    log_df = pd.DataFrame(training_history)
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    log_df.to_csv(LOG_FILE_PATH, index=False)
    print(f"Trening log sačuvan u: {LOG_FILE_PATH}")
except Exception as e:
    print(f"Greška pri čuvanju loga u CSV: {e}")