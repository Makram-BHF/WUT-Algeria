import os
import shutil
import zipfile
import sen2cor
from config import input_sen2_directory, output_sen2_directory, input_dl

input_directory = input_sen2_directory
output_directory = output_sen2_directory
if not os.path.exists(output_directory):
        os.makedirs(output_directory)
if not os.path.exists(input_dl):
        os.makedirs(input_dl)
        
# Unzip .zip files in the input directory into the output directory
def unzip_files():
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".zip"):
                zip_file = os.path.join(root, file)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(output_directory)


# Update the function to accept the output_directory and input_dl as arguments
def zip_folders():
    for folder in os.listdir(output_directory):
        folder_path = os.path.join(output_directory, folder)
        if os.path.isdir(folder_path) and "L2A" in folder:
            output_zip = os.path.join(input_dl, f"{folder}.zip")
            print(f"Output Zip Path: {output_zip}")

            # Zip the contents of the folder including the parent directory
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Calculate the relative path with respect to the parent directory
                        relative_path = os.path.relpath(file_path, output_directory)
                        zipf.write(file_path, relative_path)


if __name__ == "__main__":
    unzip_files()
    sen2cor.main()
    zip_folders()

