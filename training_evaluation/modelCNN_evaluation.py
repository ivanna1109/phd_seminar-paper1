import sys
sys.path.append('/home/ivanam/Seminar1') 

import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from training_evaluation.prepare_dataset import prepare_test_dataloader
from models.baseline_cnn import SimpleCNN 

MODEL_NAME = 'SimpleCNN'
# 👇 PUTANJA KA SAČUVANIM TEŽINAMA
SAVE_WEIGHTS_PATH = f'/home/ivanam/Seminar1/training/baseline_cnn/train_res/{MODEL_NAME}_baseline_best.pth'
# 👇 PUTANJA KA TEST PARQUET FAJLU
TEST_PARQUET = 'localWork/prepared_data/test_images.parquet' 
NUM_CLASSES = 2
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_loader, test_size = prepare_test_dataloader(parquet_path=TEST_PARQUET, batch_size=BATCH_SIZE)

try:
    baseline_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    baseline_model.load_state_dict(torch.load(SAVE_WEIGHTS_PATH, map_location=device))
    
    baseline_model.eval()
    print(f"Težine modela {MODEL_NAME} uspešno učitane.")

except Exception as e:
    print(f"Greška pri učitavanju modela: {e}")
    sys.exit()

all_preds = []
all_labels = []
total_test_loss = 0
test_batches = 0

# 👇 FIX 1: Koristimo standardni Loss za test evaluaciju
criterion = nn.CrossEntropyLoss().to(device) 

print(f"\n--- Pokretanje Evaluacije na Test Skupu ({test_size} uzoraka) ---")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = baseline_model(inputs)
        
        # 1. Računanje Loss-a
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        test_batches += 1
        
        # 2. Računanje predikcija
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = total_test_loss / test_batches

class_labels = ['Klasa 0', 'Klasa 1'] 

f1 = f1_score(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report_str = classification_report(all_labels, all_preds, target_names=class_labels)

OUTPUT_DIR = 'training_evaluation/results'
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_evaluation_results.txt')
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_content = ""
output_content += f"\n--- KONAČNA EVALUACIJA: {MODEL_NAME} ---\n"
output_content += "Evalucija modela....\n"
output_content += f"Test Loss: {test_loss:.4f}\n"
output_content += f"Test Accuracy: {accuracy:.4f}\n"
output_content += f"Test F1Score: {f1:.4f}\n"

output_content += "\n--- Classification Report ---\n"
output_content += class_report_str + "\n"
# Lepše formatiranje matrice konfuzije
output_content += "\n" + "--- Matrica Konfuzije (Confusion Matrix) ---\n"
cm_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
cm_df.index.name = 'Stvarna klasa'
cm_df.columns.name = 'Predviđena klasa'

cm_string = cm_df.to_string()
output_content += cm_string + "\n"
output_content += "Sve uspesno zavrseno.\n"

# 4. Upisivanje u fajl
try:
    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write(output_content)
    
    print(output_content) 
    print(f"\nRezultati su uspešno sačuvani u '{OUTPUT_FILE_PATH}'")

except Exception as e:
    print(f"Greška pri pisanju u fajl: {e}")