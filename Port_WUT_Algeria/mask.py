import os
import subprocess
import rasterio
from rasterio.windows import Window
import numpy as np
import numpy.ma as ma
from config import (
    clip_mask,
    mask_dir,    
    output_HR_eb,
    output_dir,
)


def process_file(file_path):
    # Rename the file
    dir_name = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    new_file_name = os.path.splitext(file_name)[0] + '_orig.tif'
    new_file_path = os.path.join(dir_name, new_file_name)
    os.rename(file_path, new_file_path)

    # Run gdalwarp command
    output_file_path = os.path.join(dir_name, file_name)
    cmd = ['gdalwarp', '-overwrite', '-cutline',
# Tunisia
#           f'{clip_mask}', '-cl', '50k-ingc',
# Algerie : mitidja
           f'{clip_mask}', '-cl', 'mitidja_aoi',
           '-crop_to_cutline', new_file_path, output_file_path]
    subprocess.run(cmd, check=True)

    # Delete the renamed file
    os.remove(new_file_path)

def process_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.tif'):
                file_path = os.path.join(root, file_name)
                process_file(file_path)

# Specify the root folder where the script will start searching

root_folder = f'{output_HR_eb}'
print ("root_folder   : ",root_folder)

# Start processing the root folder and its subfolders
process_folder(root_folder)


# Path to the mask file
mask_path = f'{mask_dir}mask_ALGERIA.tif'
# Root folder containing TIF files
root_folder = f'{output_HR_eb}'
# Output folder for masked files
output_root_folder = f'{output_HR_eb}na_masked/'

# Values to keep in the mask
mask_values_to_keep = [40, 30, 20, 10]

# Function to crop or resize array to match target shape
def crop_or_resize_array(arr, target_shape):
    if arr.shape != target_shape:
        return arr[:target_shape[0], :target_shape[1]]
    return arr

# Walk through the root folder and its subfolders
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.tif'):
            tif_path = os.path.join(root, file)
            relative_path = os.path.relpath(tif_path, root_folder)
            output_folder = os.path.join(output_root_folder, os.path.dirname(relative_path))
            os.makedirs(output_folder, exist_ok=True)

            with rasterio.open(mask_path) as mask_ds:
                mask = mask_ds.read(1)
                window = mask_ds.window(*mask_ds.bounds)

                with rasterio.open(tif_path) as tif_ds:
                    tif_band = tif_ds.read(1, window=window)

                    # Crop or resize tif_band to match mask's shape
                    tif_band = crop_or_resize_array(tif_band, mask.shape)

                    # Adjust mask shape if needed
                    mask = crop_or_resize_array(mask, tif_band.shape)

                    masked_band = np.where(np.isin(mask, mask_values_to_keep), tif_band, np.nan)
                    masked_band = ma.masked_invalid(masked_band)

                    output_filename = os.path.join(output_folder, file)

                    with rasterio.open(
                        output_filename,
                        'w',
                        driver='GTiff',
                        width=tif_band.shape[1],
                        height=tif_band.shape[0],
                        count=1,
                        dtype=masked_band.dtype,
                        crs=tif_ds.crs,
                        transform=tif_ds.transform,
                    ) as output_ds:
                        output_ds.write(masked_band.filled(np.nan), 1)


# Path to the mask file
mask_path = f'{mask_dir}mask_river_ALGERIA.tif'
root_folder = f'{output_HR_eb}na_masked/'
# Output folder for modified files
output_root_folder = f'{output_dir}masked/'

# Values to keep in the mask
mask_values_to_keep = [40, 0]

# Function to crop or resize array to match target shape
def crop_or_resize_array(arr, target_shape):
    if arr.shape != target_shape:
        return arr[:target_shape[0], :target_shape[1]]
    return arr

# Walk through the root folder and its subfolders
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith('.tif'):
            tif_path = os.path.join(root, file)
            relative_path = os.path.relpath(tif_path, root_folder)
            output_folder = os.path.join(output_root_folder, os.path.dirname(relative_path))
            os.makedirs(output_folder, exist_ok=True)

            with rasterio.open(mask_path) as mask_ds:
                mask = mask_ds.read(1)
                window = mask_ds.window(*mask_ds.bounds)

                with rasterio.open(tif_path) as tif_ds:
                    tif_band = tif_ds.read(1, window=window)

                    # Crop or resize tif_band to match mask's shape
                    tif_band = crop_or_resize_array(tif_band, mask.shape)

                    # Adjust mask shape if needed
                    mask = crop_or_resize_array(mask, tif_band.shape)

                    masked_band = np.where(np.isin(mask, mask_values_to_keep), tif_band, np.nan)
                    masked_band = ma.masked_invalid(masked_band)

                    output_filename = os.path.join(output_folder, file)

                    with rasterio.open(
                        output_filename,
                        'w',
                        driver='GTiff',
                        width=tif_band.shape[1],
                        height=tif_band.shape[0],
                        count=1,
                        dtype=masked_band.dtype,
                        crs=tif_ds.crs,
                        transform=tif_ds.transform,
                    ) as output_ds:
                        output_ds.write(masked_band.filled(np.nan), 1)


