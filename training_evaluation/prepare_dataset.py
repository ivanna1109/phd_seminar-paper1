import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import ast
from PIL import Image

class EmbryoSupervisedDataset(Dataset):
    """ Prilagođena klasa Dataseta za SimpleCNN (vraća samo sliku i labelu) """
    
    def __init__(self, data_df, transforms, minority_repeat_factor=1):
        self.data_df = data_df.reset_index(drop=True)
        self.transforms = transforms
        self.minority_repeat_factor = minority_repeat_factor
        self.effective_indices = self._create_repeated_indices()
    
    def _create_repeated_indices(self):
        # Ova logika se koristi samo za merenje efektivne veličine; 
        # za test set faktor ponavljanja je uvek 1.
        if self.data_df.empty or self.minority_repeat_factor == 1:
            return list(self.data_df.index)
        
        # Pojednostavljena logika za test set
        return list(self.data_df.index) 

    def __len__(self):
        return len(self.effective_indices)

    def __getitem__(self, idx):
        original_idx = self.effective_indices[idx]
        row = self.data_df.iloc[original_idx] 
        
        image_bytes = row['image_data']
        class_label = row['class'] 
        
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
        
        # 👇 Vraća samo 2 tenzora, jer SimpleCNN to očekuje
        return img_tensor, torch.tensor(class_label, dtype=torch.long)

def get_supervised_transforms(input_size=224):
    """ Vraća transformacije za trening i evaluaciju/test. """
    train_transforms = transforms.Compose([
        # ... (Vaše transformacije za trening) ...
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_test_transforms

def prepare_test_dataloader(parquet_path, batch_size):
    """ Funkcija za kreiranje Dataloader-a isključivo za test set. """
    test_df = pd.read_parquet(parquet_path)
    
    _, test_transforms = get_supervised_transforms(input_size=224) 
    
    test_dataset = EmbryoSupervisedDataset(
        data_df=test_df, 
        transforms=test_transforms, 
        minority_repeat_factor=1 
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    print(f"\nUčitano {len(test_df)} uzoraka za test evaluaciju.")
    return test_loader, len(test_df)