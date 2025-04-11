import os

# config atm_cor

# Set the variable for choosing the country
country_code = "ALGERIA"  # Change this to "ALGERIA" for processing ALGERIA
s2_tile = "31SDA"  # 31SDA for MITIDJA
# Define the input and output directories based on the country choice
input_sen2_directory = f"/home/karim/project/INPUT/{country_code}/sentinel2_data/"
output_sen2_directory = f"/home/karim/project/INPUT/{country_code}/sentinel2_data/{s2_tile}/RAW_S2"
# Create a directory to store the zip files
input_dl = f"/home/karim/project/scripts/deep_learning/inputs/{country_code}"
sen2cor_bat_file = "/home/karim/project/packages/Sen2Cor-02.11.00-Linux64/bin/L2A_Process"
number_of_segments = 30000
#x_pixels = 3302
#y_pixels = 2586

# config deap_learning
input_folder = f"/home/karim/project/scripts/deep_learning/inputs/{country_code}"
output_dir = f"/home/karim/project/OUTPUT/{country_code}/HR_data/"
python_interpreter = "/home/karim/miniconda3/bin/python"
rdsr_script_path = "/home/karim/project/scripts/deep_learning/RDSR.py"
output_HR_eb = f"/home/karim/project/OUTPUT/{country_code}/HR_data/extracted_bands/"

# MASK creation & application
mask_dir = '/home/karim/project/INPUT/ALGERIA/mask/'
esa_worldcover_1 = '/home/karim/project/INPUT/ALGERIA/mask/esa_worldcover/terrascope_download_20231031_103835/WORLDCOVER/ESA_WORLDCOVER_10M_2021_V200/MAP/ESA_WorldCover_10m_2021_v200_N36E000_Map/ESA_WorldCover_10m_2021_v200_N36E000_Map.tif'
clip_mask1 = '/home/karim/project/INPUT/ALGERIA/mitidja_ouest/MitidjaOuest.shp'
clip_mask = '/home/karim/project/INPUT/ALGERIA/MITIDJA_AOI/mitidja_aoi.shp'

line_file =  '/home/karim/project/INPUT/ALGERIA/mask/open_street_map/algeria-latest-free.shp/gis_osm_waterways_free_1.shp'
buffer_width = 25  # Change this value according to your requirements


























