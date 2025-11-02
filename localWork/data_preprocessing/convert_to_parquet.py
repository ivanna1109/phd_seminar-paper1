import os
import pandas as pd
from PIL import Image
import numpy as np
import sys

sys.path.append('/home/ivana-milutinovic/Documents/Doktorske/Prva godina/Seminar1/LocalWorkspace')

def load_images_to_dataframe(base_dir):
    data = []
    first_image_resolution = None
    resolutions = []
    for class_label in ['0', '1']:
        class_dir = os.path.join(base_dir, class_label)
        if not os.path.exists(class_dir):
            print(f"Upozorenje: Direktorijum '{class_dir}' ne postoji. Preskačem.")
            continue

        print(f"Učitavam slike iz: {class_dir}")
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                filepath = os.path.join(class_dir, filename)
                #print(filepath)
                try:
                    with Image.open(filepath) as img:
                        resolution = img.size
                        resolutions.append(resolution)
                        
                        if first_image_resolution is None:
                            first_image_resolution = resolution
                            print(f"   REZOLUCIJA prve slike (ŠxV): {resolution}")
                        img_array = np.array(img)
                        day = int(filename.split('_')[0][1])
                        #print(img_array.shape)
                        data.append({
                            'filename': filename,
                            'class': int(class_label),
                            'day': day,
                            'image_data': img_array.tobytes(), 
                            'image_shape': list(img_array.shape) 
                        })
                except Exception as e:
                    print(f"Greška pri učitavanju slike {filepath}: {e}")
    print(len(data))
    return pd.DataFrame(data)

def save_prepared_data(df, output_dir):
    try:
        df.to_parquet(output_dir, index=False)
        print(f"Podaci sačuvani u: {output_dir}")
    except Exception as e:
        print(e)
        print("Nema podataka za čuvanje.")

def count_classes(df, name):
    if df.empty:
        print(f"Nema podataka u {name} skupu za brojanje klasa.")
        return
        
    print(f"--- Broj klasa u {name} skupu ---")
    class_counts = df['class'].value_counts().sort_index()
    
    if 0 in class_counts:
        print(f"Klasa 0: {class_counts[0]}")
    else:
        print("Klasa 0: 0")

    if 1 in class_counts:
        print(f"Klasa 1: {class_counts[1]}")
    else:
        print("Klasa 1: 0")
    print("---------------------------------")


base_data_dir = 'raw_data'
train_dir = os.path.join(base_data_dir, 'train')
test_dir = os.path.join(base_data_dir, 'test')
train_output_dir = 'prepared_data/train_images.parquet'
test_output_dir = 'prepared_data/test_images.parquet'

print("--- Učitavanje trening podataka ---")
train_df = load_images_to_dataframe(train_dir)
print(f'Train shape {train_df.shape}')
count_classes(train_df, "trening")
#save_prepared_data(train_df, train_output_dir)
print("\n--- Učitavanje test podataka ---")
test_df = load_images_to_dataframe(test_dir)
print(f'Test shape {test_df.shape}')
count_classes(test_df, "test")
save_prepared_data(test_df, test_output_dir)


print("\nZavršeno!")