import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import ast
from PIL import Image
import torch
import sys
import math
from torch.utils.data import random_split
from torch.utils.data import DataLoader

class EmbryoSupervisedDataset(Dataset):
    
    def __init__(self, data_df, transforms, minority_repeat_factor=1):
        """
        Inicijalizacija Dataseta. Prihvata DataFrame, a ne putanju do Parquet fajla.
        """
        self.data_df = data_df.reset_index(drop=True)
        self.transforms = transforms
        self.minority_repeat_factor = minority_repeat_factor
        
        self.effective_indices = self._create_repeated_indices()
        
        print(f"Dataset inicijalizovan. Originalna veličina: {len(self.data_df)}. Efektivna veličina: {len(self.effective_indices)}.")

    def _create_repeated_indices(self):
        """
        Ponavlja indekse manjinske klase (bazirano na koloni 'class').
        (Metoda ostaje nepromenjena)
        """
        if self.data_df.empty:
            return []
            
        class_counts = self.data_df['class'].value_counts()
        
        if len(class_counts) < 2:
            return list(self.data_df.index)
            
        minority_class = class_counts.index[-1]
        final_indices = []
        
        for idx in self.data_df.index:
            class_label = self.data_df.loc[idx, 'class']
            
            if class_label == minority_class:
                final_indices.extend([idx] * self.minority_repeat_factor)
            else:
                final_indices.append(idx)
                
        return final_indices

    def __len__(self):
        return len(self.effective_indices)

    def __getitem__(self, idx):
        
        original_idx = self.effective_indices[idx]
        row = self.data_df.iloc[original_idx] 
        
        image_bytes = row['image_data']
        class_label = row['class'] 
        
        # 👇 KLJUČNA IZMENA 1: Ekstrakcija dana embrija
        day_value = row['day']
        
        # 💡 Napomena: Dan se mora konvertovati u float32 tenzor (i biti vektor dimenzije 1)
        # Model EmbryoNet2 normalizuje ovo interno, tako da je slanje sirove vrednosti dana (3 ili 5) OK.
        day_tensor = torch.tensor([day_value], dtype=torch.float32) 
        
        # ... (Ostatak koda za obradu slike ostaje isti) ...
        img_shape = row['image_shape']
        if isinstance(img_shape, bytes):
            img_shape = img_shape.decode('utf-8') 
        if isinstance(img_shape, str):
            try:
                img_shape = ast.literal_eval(img_shape)
            except:
                img_shape = [224, 224, 3]
        if not isinstance(img_shape, tuple):
             img_shape = tuple(img_shape)
        
        image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape(img_shape)

        if image_array.ndim == 2:
            img = Image.fromarray(image_array, mode='L').convert('RGB')
        else:
            img = Image.fromarray(image_array)

        img_tensor = self.transforms(img)
        
        return img_tensor, day_tensor, torch.tensor(class_label, dtype=torch.long)
    
def get_supervised_transforms(input_size=224):
    """
    Definisanje standardnih transformacija za nadzirano učenje (finetuning ili baseline trening).
    Koristi blage augmentacije za generalizaciju.
    """
    # Augmentacije koje čuvaju relevantne embriološke feature
    train_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10), # Blaga rotacija
        # Manje agresivne promene kontrasta/osvetljenja nego kod SSL
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        # ImageNet Normalizacija (standard za modele bazirane na CNN)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # Transformacije za Validaciju/Test (SAMO skaliranje i normalizacija)
    val_test_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms


def prepare_supervised_dataloaders(parquet_path, num_classes, val_split_ratio, batch_size, minority_repeat_train, input_size=224): 
    try:
        train_transforms, val_transforms = get_supervised_transforms(input_size)
    except NameError:
        print("Greška: get_supervised_transforms funkcija nije definisana.")
        sys.exit()

    # 1. Učitavanje celog skupa
    full_df = pd.read_parquet(parquet_path)
    
    
    total_size = len(full_df)
    val_size = int(total_size * val_split_ratio)
    train_size = total_size - val_size
    
    # Privremena podela indeksa (koristiti stratifikovanu podelu ako je moguće)
    g = torch.Generator().manual_seed(42)
    train_indices_set, val_indices_set = random_split(range(total_size), [train_size, val_size], generator=g)
    
    train_df = full_df.iloc[train_indices_set.indices].reset_index(drop=True)
    val_df = full_df.iloc[val_indices_set.indices].reset_index(drop=True)
    
    # --- PROVERA BALANSA TRENING SKUPA ---
    train_class_counts = train_df['class'].value_counts().sort_index()
    
    if len(train_class_counts) != num_classes:
        print(f"Greška: Pronađeno {len(train_class_counts)} klasa umesto {num_classes}.")
        sys.exit()

    # Pretpostavimo da je indeks 0 većinska klasa, indeks 1 manjinska.
    majority_count = train_class_counts.max()
    minority_count = train_class_counts.min()

    # 3. Dinamički proračun faktora ponavljanja i težina gubitka
    
    # Računanje optimalnog faktora za 1:1 oversampling
    optimal_repeat_factor = math.ceil(majority_count / minority_count)
    
    # Ako je NUM_REPEATS_TRAIN već definisan (npr. 5), koristi se on.
    # Ako je prenet 1 (što je default), onda treba koristiti optimalan. 
    # U ovom slučaju, koristimo prenetu vrednost, ali ispisujemo optimalnu.
    
    print(f"\n--- Analiza Trening Skupa (Originalno) ---")
    print(f"Klasa {train_class_counts.idxmax()} (Većinska): {majority_count}")
    print(f"Klasa {train_class_counts.idxmin()} (Manjinska): {minority_count}")
    print(f"Koristi se faktor ponavljanja (minority_repeat_train): {minority_repeat_train} (Optimalno: {optimal_repeat_factor})")
    
    # 💡 NOVI KORAK: Izračunavanje težina gubitka (WEIGHTED LOSS)
    
    total_train_samples = majority_count + minority_count
    
    # Izračunavamo težine bazirane na originalnom, neuravnoteženom trening setu
    # Težina = Ukupno_uzoraka / (Broj_klasa * Broj_uzoraka_te_klase)
    weight_0 = total_train_samples / (num_classes * train_class_counts.get(0, 1)) 
    weight_1 = total_train_samples / (num_classes * train_class_counts.get(1, 1)) 
    
    # Kreiranje tenzora težina
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float)
    print(f"Izračunate težine klase (za CrossEntropyLoss, Klasa 0:Klasa 1): {weight_0:.2f}:{weight_1:.2f}")

    # 4. Kreiranje finalnih Dataset objekata
    try:
        train_dataset_final = EmbryoSupervisedDataset(
            data_df=train_df, 
            transforms=train_transforms, 
            minority_repeat_factor=minority_repeat_train # Koristimo uneti faktor
        )
        val_dataset_final = EmbryoSupervisedDataset(
            data_df=val_df, 
            transforms=val_transforms, 
            minority_repeat_factor=1
        )
    except NameError:
        print("Greška: EmbryoSupervisedDataset klasa nije definisana.")
        sys.exit()

    # 5. Kreiranje DataLoadera
    train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"\nTRENING SET (Efektivna veličina, oversampling): {len(train_dataset_final)} uzoraka.")
    print(f"VALIDACIONI SET (Originalna veličina): {len(val_dataset_final)} uzoraka.")
    
    # 6. Povratak DataLoadera I TEŽINA
    return train_loader, val_loader, class_weights

def prepare_test_dataloader(parquet_path, batch_size, input_size=224):
    """ Funkcija za kreiranje Dataloader-a isključivo za test set za EmbryoNet2 (3 ulaza). """
    test_df = pd.read_parquet(parquet_path)
    
    # Koristimo transformacije za validaciju/test (bez augmentacije)
    _, test_transforms = get_supervised_transforms(input_size) 
    
    # Kreiranje Dataseta (faktor ponavljanja=1 za test set)
    test_dataset = EmbryoSupervisedDataset(
        data_df=test_df, 
        transforms=test_transforms, 
        minority_repeat_factor=1 
    )
    
    # Kreiranje Dataloader-a (shuffle=False je OBAVEZNO za evaluaciju)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"\nUčitano {len(test_df)} uzoraka za test evaluaciju (EmbryoNet2 format - slika, dan, labela).")
    return test_loader, len(test_df)