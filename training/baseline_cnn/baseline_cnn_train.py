import sys

sys.path.append('/home/ivanam/Seminar1')
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
import os
import pandas as pd
from models.baseline_cnn import SimpleCNN 
from torch.optim.lr_scheduler import CosineAnnealingLR
from prepare_data import prepare_supervised_dataloaders

# --- KONFIGURACIJA ---
MODEL_NAME = 'SimpleCNN' 
NUM_CLASSES = 2
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# PROMENA: TRAIN_PARQUET će se podeliti na Train/Val. TEST_PARQUET ostaje za finalni test.
TRAIN_PARQUET = 'localWork/prepared_data/train_images.parquet' 
VAL_SPLIT_RATIO = 0.2 # 20% podataka ide u validacioni set
SAVE_PATH = f'/home/ivanam/Seminar1/training/baseline_cnn/train_res/{MODEL_NAME}_baseline_best.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nKoristi se uređaj: {device}")
NUM_REPEATS_TRAIN = 5
try:
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

# --- 2. INICIJALIZACIJA MODELA (Ostatak koda ostaje isti) ---
baseline_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

# --- 3. OPTIMIZATOR I GUBITAK ---
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 
optimizer = Adam(baseline_model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# --- 4. TRENING PETLJA ---

print(f"\n--- Pokretanje Supervised Treninga za Baseline model {MODEL_NAME} ---")
best_val_f1 = 0.0
training_history = []
LOG_FILE_PATH = f'training/baseline_cnn/train_res/{MODEL_NAME}_training_log.csv'
for epoch in range(1, NUM_EPOCHS + 1):
    baseline_model.train()
    total_loss = 0
    train_preds = []
    train_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = baseline_model(inputs)
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
    baseline_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = baseline_model(inputs)
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
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(baseline_model.state_dict(), SAVE_PATH)
        print("   --> Sačuvan najbolji Baseline model!")
    scheduler.step()

print("\n--- Supervised Trening završen. ---")
try:
    log_df = pd.DataFrame(training_history)
    log_df.to_csv(LOG_FILE_PATH, index=False)
    print(f"Trening log sačuvan u: {LOG_FILE_PATH}")
except Exception as e:
    print(f"Greška pri čuvanju loga u CSV: {e}")