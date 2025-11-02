import sys
sys.path.append('/home/ivanam/Seminar1') 
import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from torchvision.models import efficientnet_b0
from training_evaluation.prepare_dataset import prepare_test_dataloader 
from training.efficientNet.prepare_model import clean_state_dict_keys 


MODEL_NAME = 'efficientnet_b0_finetuned'
NUM_CLASSES = 2
BATCH_SIZE = 32 
SAVE_WEIGHTS_PATH = f'/home/ivanam/Seminar1/training/finetuned_models/eNet/{MODEL_NAME}_best_model.pth' 
TEST_PARQUET = 'localWork/prepared_data/test_images.parquet' 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = 'training_evaluation/results' # Folder za rezultate
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_evaluation_results.txt')

# 1. Priprema Test Podataka
test_loader, test_size = prepare_test_dataloader(parquet_path=TEST_PARQUET, batch_size=BATCH_SIZE)

# 2. Inicijalizacija Modela i Učitavanje Najboljih Težina
try:
    # Inicijalizacija EfficientNet-B0 bez ImageNet težina
    model = efficientnet_b0(weights=None) 
    
    # Prilagođavanje klasifikacionog sloja na 2 klase
    in_features = model.classifier[1].in_features 
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    
    # Učitavanje finetuniranih težina
    model.load_state_dict(torch.load(SAVE_WEIGHTS_PATH, map_location=device))
    
    model.to(device)
    model.eval()
    print(f"Težine modela {MODEL_NAME} uspešno učitane i model je spreman za evaluaciju.")

except Exception as e:
    print(f"Greška pri učitavanju modela {MODEL_NAME}: {e}")
    sys.exit()

all_preds = []
all_labels = []
total_test_loss = 0
test_batches = 0

criterion = nn.CrossEntropyLoss().to(device) 

print(f"\n--- Pokretanje Evaluacije na Test Skupu ({test_size} uzoraka) ---")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        total_test_loss += loss.item()
        test_batches += 1
        
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = total_test_loss / test_batches

class_labels = ['Klasa 0', 'Klasa 1'] 

f1 = f1_score(all_labels, all_preds, average='binary')
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report_str = classification_report(all_labels, all_preds, target_names=class_labels)

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_content = ""
output_content += f"\n--- KONAČNA EVALUACIJA: {MODEL_NAME} ---\n"
output_content += "Evalucija modela....\n"
output_content += f"Test Loss: {test_loss:.4f}\n"
output_content += f"Test Accuracy: {accuracy:.4f}\n"
output_content += f"Test F1Score: {f1:.4f}\n"

# Izveštaj Klasifikacije
output_content += "\n--- Classification report ---\n"
output_content += class_report_str + "\n"

# Matrica Konfuzije
output_content += "\n" + "--- Matrica Konfuzije (Confusion matrix) ---\n"
cm_df = pd.DataFrame(conf_matrix, index=class_labels, columns=class_labels)
cm_df.index.name = 'Stvarna Klasa'
cm_df.columns.name = 'Predviđena Klasa'

cm_string = cm_df.to_string()
output_content += cm_string + "\n"
output_content += "Sve uspesno zavrseno.\n"

# Upisivanje u fajl
try:
    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write(output_content)
    
    print(output_content) 
    print(f"\nRezultati su uspešno sačuvani u '{OUTPUT_FILE_PATH}'")

except Exception as e:
    print(f"Greška pri pisanju u fajl: {e}")