import os
import cv2

def flip_images_in_subfolders(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Traverse the directory tree
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # Check if the file is an image (you can add more extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)

                try:
                    # Read the image
                    img = cv2.imread(file_path)

                    # Flip the image horizontally
                    flipped_img = cv2.flip(img, 1)

                    # Construct the output path
                    relative_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save the flipped image
                    output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_flipped{os.path.splitext(file)[1]}")
                    cv2.imwrite(output_path, flipped_img)
                    print(f"Flipped image saved as {output_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Example usage
input_folder = r"D:\Deaf and Dumb\dataset\f"
output_folder = r"D:\Deaf and Dumb\dataset"
flip_images_in_subfolders(input_folder, output_folder)
