import os
import shutil

# Define the source and destination directories
source_dir = r"D:\Deaf and Dumb\archive (3)\asl_alphabet_train\asl_alphabet_train"
destination_dir = r"D:\Deaf and Dumb\New folder"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Iterate over each subfolder in the source directory
for subfolder in os.listdir(source_dir):
    subfolder_path = os.path.join(source_dir, subfolder)
    
    # Check if it's a directory
    if os.path.isdir(subfolder_path):
        # Create a corresponding subfolder in the destination directory
        dest_subfolder_path = os.path.join(destination_dir, subfolder)
        os.makedirs(dest_subfolder_path, exist_ok=True)
        
        # List all files in the subfolder
        files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
        
        # Process only the first 200 files
        for filename in files[:200]:
            old_file = os.path.join(subfolder_path, filename)
            new_file = os.path.join(dest_subfolder_path, filename)
            
            # Move the file to the new subfolder
            shutil.move(old_file, new_file)
            print(f'Moved: {filename} to {dest_subfolder_path}')
