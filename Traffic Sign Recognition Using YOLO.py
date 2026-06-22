import os
import pandas as pd
import shutil
import subprocess
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- 1. SETUP AND DIRECTORIES ---
def setup_drive():
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        os.chdir('/content/gdrive/My Drive/Colab Notebooks/traffic')
        print(f"Working in: {os.getcwd()}")
    except ImportError:
        print("Not in Colab. Ensure your local paths are correct.")

# --- 2. BUILD DARKNET ---
def build_darknet():
    if not os.path.exists('darknet'):
        subprocess.run(["git", "clone", "https://github.com/AlexeyAB/darknet.git"], check=True)
    
    os.chdir('darknet')
    # Modify Makefile for GPU and OpenCV
    subprocess.run(["sed", "-i", "s/OPENCV=0/OPENCV=1/", "Makefile"], check=True)
    subprocess.run(["sed", "-i", "s/GPU=0/GPU=1/", "Makefile"], check=True)
    subprocess.run(["sed", "-i", "s/CUDNN=0/CUDNN=1/", "Makefile"], check=True)
    subprocess.run(["sed", "-i", "s/CUDNN_HALF=0/CUDNN_HALF=1/", "Makefile"], check=True)
    
    # Build
    subprocess.run(["make"], check=True)
    os.chdir('..')

# --- 3. DATA PREPROCESSING ---
def preprocess_data(os_dir, tfinal_dir):
    if not os.path.exists(tfinal_dir):
        os.makedirs(tfinal_dir)

    # Note: Using pd.concat instead of .append as .append is deprecated in newer Pandas
    all_csv_list = []

    for folder in os.listdir(os_dir):
        if folder == '.DS_Store':
            continue
        
        inner_dir = os.path.join(os_dir, folder)
        print(f"Processing folder: {folder}")
        
        # Rename and update CSV files
        for item in os.listdir(inner_dir):
            if item == f"GT-{folder}.csv":
                csv_path = os.path.join(inner_dir, item)
                csv_file = pd.read_csv(csv_path, sep=';')
                csv_file['Filename'] = csv_file['Filename'].apply(lambda x: f'{folder}_{x}')
                
                new_csv_name = f'GGT-{folder}.csv'
                csv_file.to_csv(os.path.join(inner_dir, new_csv_name), sep=';', index=False)
                all_csv_list.append(csv_file)
            
            # Copy images to TFinal
            elif item.endswith('.ppm'):
                # Note: The original notebook had a 'continue' before the rename logic. 
                # I am assuming you want to copy the images to the central folder.
                shutil.copy(os.path.join(inner_dir, item), tfinal_dir)

    train_csv = pd.concat(all_csv_list, ignore_index=True)
    return train_csv

# --- 4. YOLO ANNOTATION HELPERS ---
def convert_yolo_labels(row):
    """
    Converts ROI coordinates to YOLO format:
    [class] [x_center] [y_center] [width] [height]
    """
    dw = 1.0 / row['Width']
    dh = 1.0 / row['Height']
    
    # Center coordinates
    x = (row['Roi.X1'] + row['Roi.X2']) / 2.0
    y = (row['Roi.Y1'] + row['Roi.Y2']) / 2.0
    
    # Width and Height of the box
    w = abs(row['Roi.X2'] - row['Roi.X1'])
    h = abs(row['Roi.Y2'] - row['Roi.Y1'])
    
    return x * dw, y * dh, w * dw, h * dh

def generate_annotations(train_csv, tfinal_dir):
    os.chdir(tfinal_dir)
    for _, row in train_csv.iterrows():
        txt_name = row['Filename'].replace('.ppm', '.txt')
        x, y, w, h = convert_yolo_labels(row)
        
        with open(txt_name, 'w+') as f:
            f.write(f"{row['ClassId']} {x} {y} {w} {h}")
    os.chdir('..')

# --- 5. CONFIGURATION GENERATION ---
def create_config_files():
    classes = [
        'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 
        'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
        'Speed limit (100km/h)', 'Speed limit (120km/h)', 'No passing',
        'No passing for vechiles over 3.5 metric tons', 'No vehicles',
        'Vechiles over 3.5 metric tons prohibited', 'Right-of-way at the next intersection',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
        'Priority road', 'Yield', 'Stop', 'No entry', 'End of speed limit (80km/h)',
        'End of all speed and passing limits', 'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons'
    ]

    with open('classes.names', 'w+') as f:
        for c in classes:
            f.write(f"{c}\n")

    config = {
        'classes': 43,
        'train': 'train.txt',
        'valid': 'test.txt',
        'names': 'classes.names',
        'backup': 'backup'
    }
    with open('labelled_data.data', 'w+') as f:
        for key, value in config.items():
            f.write(f"{key} = {value}\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_drive()
    # build_darknet() # Uncomment to build on first run
    
    OS_DIR = 'GTSRB/Final_Training/Images/'
    TFINAL_DIR = 'TFinal'
    
    # Process data
    full_train_csv = preprocess_data(OS_DIR, TFINAL_DIR)
    generate_annotations(full_train_csv, TFINAL_DIR)
    
    # Split
    train_ds, valid_ds = train_test_split(
        full_train_csv, test_size=0.3, random_state=43, 
        shuffle=True, stratify=full_train_csv['ClassId']
    )
    
    # Write path files
    with open('train.txt', 'w+') as f:
        for name in train_ds['Filename']:
            f.write(f"TFinal/{name}\n")
            
    with open('test.txt', 'w+') as f:
        for name in valid_ds['Filename']:
            f.write(f"TFinal/{name}\n")
            
    create_config_files()

    print("Setup complete. You can now run the Darknet training command.")
    # subprocess.run(["./darknet/darknet", "detector", "train", "labelled_data.data", ...])