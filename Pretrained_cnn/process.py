import os

# Define the fixed class-to-label mapping
CLASS_TO_LABEL = {
    "Healthy": 0,
    "Unhealthy": 1
}

def process_and_save_labels(dataset_dir):
    """
    Process dataset folders (train, valid, test), assign fixed numeric labels to images,
    and save the image paths with corresponding labels in text files (without leading `./`).
    
    Args:
    - dataset_dir (str): The root directory of the dataset.
    """
    
    folder_names = ['train', 'valid', 'test']  

    for folder_name in folder_names:
        folder_path = os.path.join(dataset_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"Error: The folder {folder_name} does not exist in the dataset.")
            continue

        output_file = f"{folder_name}_labels.txt"

        with open(output_file, 'w') as file:
            found_images = False  
            
            for class_name, class_index in CLASS_TO_LABEL.items():
                class_path = os.path.join(folder_path, class_name)

                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)

                        if img_name.lower().endswith('.jpg'):
                            # Convert to relative path without `./`
                            relative_path = os.path.relpath(img_path, dataset_dir)

                            # Write the clean path without `./`
                            file.write(f"/plantvillage_dataset/{relative_path}, {class_index}\n")
                            found_images = True  

            if not found_images:
                print(f"Warning: No images found in {folder_name}")

        print(f"Labels saved in {output_file}")

# Example usage:
dataset_dir = "./plantvillage_dataset"
process_and_save_labels(dataset_dir)