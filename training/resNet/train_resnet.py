import sys
sys.path.append('/home/ivanam/Seminar1')
import torch
import torch.nn as nn
from torchvision.models import resnet18
import os
from training.efficientNet.prepare_model import clean_state_dict_keys
from training.baseline_cnn.prepare_data import prepare_supervised_dataloaders
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

# --- PODEŠAVANJA ---
# Promenite ovo u zavisnosti od lokacije vašeg .pth fajla i broja klasa
PATH_TO_EFFICIENTNET_WEIGHTS = '/home/ivanam/Seminar1/pretraining/ssl_checkpoints/resnet18_embryo_sslbs128_best.pth'
NUM_CLASSES = 2 

# 2. Učitavanje vaših sačuvanih težina (.pth fajl)
model = resnet18(weights=None) 

# 2. Učitavanje i čišćenje state_dict
try:
    print(f"Učitavanje težina iz: {PATH_TO_EFFICIENTNET_WEIGHTS}")
    
    # Učitavanje 'state_dict' objekta
    state_dict = torch.load(PATH_TO_EFFICIENTNET_WEIGHTS, map_location=torch.device('cpu'))

    # Provera da li je state_dict u nekom ključu (checkpoint)
    if 'state_dict' in state_dict:
        original_state_dict = state_dict['state_dict']
    elif 'model' in state_dict:
        original_state_dict = state_dict['model']
    else:
        original_state_dict = state_dict

    # --- KLJUČNI KORAK: PREIMENOVANJE KLJUČEVA ---
    cleaned_state_dict = clean_state_dict_keys(original_state_dict)

    model.load_state_dict(cleaned_state_dict, strict=False) 
    print("Težine EfficientNet-B0 su uspešno učitane (sa preimenovanjem ključeva).")

except Exception as e:
    print(f"FATALNA GREŠKA pri učitavanju težina: {e}")
    print("Preimenovanje ključeva nije uspelo. Možda je potrebna specifičnija funkcija mapiranja.")

in_features = model.fc.in_features 

# Zamena poslednjeg linearnog sloja novim.
# ResNet koristi atribut 'fc' za klasifikacioni sloj.
model.fc = nn.Linear(in_features, NUM_CLASSES) 

print(f"Klasifikacioni sloj ResNet18 je modifikovan na {NUM_CLASSES} izlaznih klasa.")

BATCH_SIZE = 32
# Učitavanje dataloadera i class weights
train_loader, val_loader, class_weights = prepare_supervised_dataloaders(
    parquet_path='localWork/prepared_data/train_images.parquet',
    num_classes=NUM_CLASSES,
    val_split_ratio=0.2,
    batch_size=BATCH_SIZE,
    minority_repeat_train=5 # ILI koristite faktor oversamplinga (npr. 3, 5, itd.)
)

# Kriterijum (Loss) - PREMESTITE GA NA GPU/CPU
# Prosleđivanje težina direktno u CrossEntropyLoss
LEARNING_RATE = 3e-4 # Povećano
WEIGHT_DECAY = 1e-4  # Dodato

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM) 
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kriterijum (Loss): {criterion.__class__.__name__}")
print(f"Optimizator: {optimizer.__class__.__name__} sa lr={LEARNING_RATE}")
print(f"Scheduler: {scheduler.__class__.__name__} (smanjuje LR tokom treninga)")
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

NUM_EPOCHS = 75
MODEL_NAME = 'resnet18_ssl_finetuned' 
SAVE_PATH = f'training/finetuned_models/resNet/{MODEL_NAME}_best_model.pth' 
# --------------------------------------------------

# Postavljanje modela na pravu putanju (GPU/CPU)
model.to(device)

print(f"\n--- Pokretanje Supervised Finetuninga za model {MODEL_NAME} ---")
best_val_f1 = 0.0
training_history = []
PATIENCE = 20        # Broj epoha koje će čekati bez poboljšanja
patience_counter = 0 # Inicijalizacija brojača
LOG_FILE_PATH = f'training/resNet/results/{MODEL_NAME}_training_log.csv'

for epoch in range(1, NUM_EPOCHS + 1):
    # FAZA TRENINGA
    model.train()
    total_loss = 0
    train_preds = []
    train_labels = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        train_preds.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
    avg_train_loss = total_loss / len(train_loader)
    
    # Koristite 'binary' za binarnu klasifikaciju (dve klase)
    train_f1 = f1_score(train_labels, train_preds, average='binary') 
    train_accuracy = accuracy_score(train_labels, train_preds)
    
    # VALIDACIONA FAZA
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

    # Čuvanje najboljeg modela (na osnovu validacionog F1)
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0  # Resetovanje brojača, jer je došlo do poboljšanja
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH) 
        print("   --> Sačuvan najbolji Finetuned model!")
    else:
        patience_counter += 1 # Povećanje brojača, nije bilo poboljšanja
        print(f"   --> Nema poboljšanja. Patience brojač: {patience_counter}/{PATIENCE}")

    scheduler.step() # Podešavanje stope učenja

    # KLJUČNI DEO: Provera ranog zaustavljanja
    if patience_counter >= PATIENCE:
        print(f"\n! RANO ZAUSTAVLJANJE ! Validacioni F1 se nije poboljšao {PATIENCE} epoha.")
        break # Prekid petlje za trening

print("\n--- Finetuning završen. ---")
try:
    log_df = pd.DataFrame(training_history)
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    log_df.to_csv(LOG_FILE_PATH, index=False)
    print(f"Trening log sačuvan u: {LOG_FILE_PATH}")
except Exception as e:
    print(f"Greška pri čuvanju loga u CSV: {e}")