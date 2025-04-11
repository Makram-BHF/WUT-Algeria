import os
import glob
import subprocess
import shutil
from osgeo import gdal
from config import (
    output_sen2_directory,
    clip_mask,
    input_folder,
    output_dir,
    python_interpreter,
    rdsr_script_path,
    country_code,
    output_HR_eb,
)


def gdal_warp_and_process(jp2_file, cutline_shp, output_tif):
    # Run gdalwarp command
    gdalwarp_command = f"gdalwarp -overwrite -s_srs EPSG:32631 -t_srs EPSG:32631 -of GTiff -cutline {cutline_shp} -cl mitidja_aoi -crop_to_cutline {jp2_file} {output_tif}"
    subprocess.run(gdalwarp_command, shell=True)

    # Open the TIF file using GDAL
    tif_dataset = gdal.Open(output_tif)

    # Get image dimensions
    tif_width = tif_dataset.RasterXSize
    tif_height = tif_dataset.RasterYSize

    intersection_info = {
        "intersection_width": tif_width,
        "intersection_height": tif_height
    }

    # Get geotransform information of both files
    jp2_dataset = gdal.Open(jp2_file)
    jp2_geo_transform = jp2_dataset.GetGeoTransform()
    tif_geo_transform = tif_dataset.GetGeoTransform()

    # Calculate the corresponding pixel in the JP2 file that matches the TIF top-left coordinates
    jp2_pixel_x = int((tif_geo_transform[0] - jp2_geo_transform[0]) / jp2_geo_transform[1])
    jp2_pixel_y = int((tif_geo_transform[3] - jp2_geo_transform[3]) / jp2_geo_transform[5])

    corresponding_pixel = {
        "jp2_pixel_x": jp2_pixel_x,
        "jp2_pixel_y": jp2_pixel_y
    }

    return intersection_info, corresponding_pixel


def execute_rdsr_on_zip(zip_file):
    if os.access(python_interpreter, os.X_OK) and os.path.isfile(rdsr_script_path):
        print(f'Executing: {python_interpreter} {rdsr_script_path} {zip_file} --x {first_x} --y {first_y} --size "{y_pixels},{x_pixels}"')
        os.system(f'{python_interpreter} {rdsr_script_path} {zip_file} --x {first_x} --y {first_y} --size "{y_pixels},{x_pixels}"')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for file in os.listdir():
            if (file.startswith("S2A") or file.startswith("S2B")) and file.endswith(".tif"):
                shutil.move(file, output_dir)
        print(f"Execution completed for: {zip_file}")
    else:
        print("Python interpreter or RDSR.py script not found.")

def extract_bands(input_tif_file, output_HR_eb):
    dataset = gdal.Open(input_tif_file, gdal.GA_ReadOnly)

    if dataset is None:
        print(f"Error: Could not open {input_tif_file}")
        return

    num_bands = dataset.RasterCount
    subfolder_name = os.path.splitext(os.path.basename(input_tif_file))[0][11:19]

    output_dir_eb = os.path.join(output_HR_eb, subfolder_name)
    if not os.path.exists(output_dir_eb):
        os.makedirs(output_dir_eb)

    for band_number in range(1, num_bands + 1):
        band = dataset.GetRasterBand(band_number)
        band_name = band.GetDescription() or f"band{band_number}"
        output_file = os.path.join(output_dir_eb, f"{band_name}.tif")
        gdal.Translate(output_file, dataset, format='GTiff', bandList=[band_number])

    dataset = None

def process_zip_files():
    for zip_file in os.listdir(input_folder):
        if zip_file.endswith(".zip"):
            print(f"Processing zip file: {zip_file}")
            execute_rdsr_on_zip(os.path.join(input_folder, zip_file))

def run_extract_bands_script():
    tif_files = [file for file in os.listdir(output_dir) if file.endswith(".tif")]

    for tif_file in tif_files:
        input_tif_file = os.path.join(output_dir, tif_file)
        extract_bands(input_tif_file, output_HR_eb)


# Define the path pattern
path_pattern = f'{output_sen2_directory}/*L2A*/*/L2A*/*/*/*B04_10m.jp2'

# Use glob to find files matching the pattern
matching_files = glob.glob(path_pattern)

# If files are found, select the first file
if matching_files:
    first_file = matching_files[0]
    print(f"First file matching the criteria: {first_file}")
else:
    print("No file found matching the criteria.")


output_tif = f'{output_sen2_directory}/intesection.tif'

intersection_info, corresponding_pixel = gdal_warp_and_process(first_file, clip_mask, output_tif)
os.remove(output_tif)
print("Intersection:")
for key, value in intersection_info.items():
    print(f"{key}: {value}")

print(f"Top-left pixel of TIF file in JP2 file: (X: {corresponding_pixel['jp2_pixel_x']}, Y: {corresponding_pixel['jp2_pixel_y']})")

# Given intersection width and height
intersection_width = intersection_info["intersection_width"]
intersection_height = intersection_info["intersection_height"]
#intersection_width = 100
#intersection_height = 50

    
# Calculate 3/4 of the larger value
three_fourths_of_larger = max(intersection_width, intersection_height)

# Check if the smaller value is less than 3/4 of the larger one
if min(intersection_width, intersection_height) < three_fourths_of_larger:
    # Adjust the smaller value to be 3/4 of the larger value
    smaller_adjusted = int(three_fourths_of_larger)
    
    # Assign the adjusted value to the smaller dimension
    if intersection_width < intersection_height:
        intersection_width = smaller_adjusted
    else:
        intersection_height = smaller_adjusted


# Calculate the next multiple of 8 for intersection_width
#if intersection_width % 360 != 0:
#    intersection_width = ((intersection_width // 360) + 1) * 360

# Calculate the next multiple of 8 for intersection_height
#if intersection_height % 360 != 0:
#    intersection_height = ((intersection_height // 360) + 1) * 360


# Now, intersection_width and intersection_height hold the adjusted values
print(f"Adjusted Width: {intersection_width}, Adjusted Height: {intersection_height}")



x_pixels = intersection_width
y_pixels = intersection_height
print(f'x_pixels: {x_pixels}, y_pixels: {y_pixels}')


first_x = corresponding_pixel["jp2_pixel_x"]
first_y = corresponding_pixel["jp2_pixel_y"]
print(f'first_x: {first_x}, first_y: {first_y}')

print(f"Executing:  --x {first_x} --y {first_y} --size {x_pixels} {y_pixels}")


if __name__ == "__main__":
    print("Starting RDSR.py execution...")
    process_zip_files()
    print("RDSR.py execution completed.")

    print("Starting extract_bands.py execution...")
    run_extract_bands_script()
    print("Script execution completed.")

