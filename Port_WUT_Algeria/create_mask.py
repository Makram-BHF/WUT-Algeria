import os
import rasterio
from osgeo import gdal, ogr
import geopandas as gpd
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon
from config import (
    mask_dir,
    esa_worldcover_1,
    clip_mask,
    line_file,
    buffer_width,
)


# Function to clip to the AOI using GDAL
def clip_to_AOI(input_file, output_file, cutline_shp):
    warp_options = gdal.WarpOptions(
        cutlineDSName=cutline_shp,
        cropToCutline=True,
        dstNodata=-9999.0
    )
    gdal.Warp(output_file, input_file, options=warp_options)


# Function to merge ESA WorldCover files using GDAL
def merge_esa_files(output_file, input_files_list):
    gdal_merge_cmd = ['gdal_merge.py', '-ot', 'Float32', '-of', 'GTiff', '-o', output_file, '--optfile',
                      input_files_list]
    os.system(' '.join(gdal_merge_cmd))


# Function to reproject the mask using GDAL
def reproject_mask(input_file, output_file, target_srs):
    gdal.Warp(output_file, input_file, dstSRS=target_srs)


def create_buffered_polygons(input_file, output_file, buffer_width, epsg_code):
    # Load the line shapefile
    lines = gpd.read_file(input_file)

    # Reproject the GeoDataFrame to a projected CRS
    lines = lines.to_crs(epsg=epsg_code)

    # Create a buffer around the lines with the specified width
    polygons = lines.buffer(buffer_width)

    # Create a GeoDataFrame from the polygons
    polygon_gdf = gpd.GeoDataFrame(geometry=polygons, crs=lines.crs)

    # Save the polygon GeoDataFrame to a new shapefile
    polygon_gdf.to_file(output_file)


def reproject_mask_ogr(input_path, output_path, target_srs):
    # Open the input shapefile
    input_ds = ogr.Open(input_path)
    if input_ds is None:
        print(f"Failed to open {input_path}")
        return

    # Get the input layer
    input_layer = input_ds.GetLayer()

    # Create output spatial reference
    output_srs = ogr.osr.SpatialReference()
    output_srs.ImportFromEPSG(target_srs)

    # Create output layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_ds = driver.CreateDataSource(output_path)
    output_layer = output_ds.CreateLayer('reprojected_mask', srs=output_srs)

    # Reproject features
    for feature in input_layer:
        geometry = feature.GetGeometryRef()
        geometry.TransformTo(output_srs)
        new_feature = ogr.Feature(output_layer.GetLayerDefn())
        new_feature.SetGeometry(geometry)
        output_layer.CreateFeature(new_feature)
        new_feature = None

    # Close datasets
    input_ds = None
    output_ds = None


# Function to clip input.shp using reprojected mask using OGR
def clip_input_using_mask(input_shp, clip_mask, output_shp):
    cmd = f'ogr2ogr -clipsrc {clip_mask} {output_shp} {input_shp}'
    os.system(cmd)


# Function to clip using gdalwarp
def clip_using_gdalwarp(input_tif, output_tif, cutline_shp):
    gdal.Warp(output_tif, input_tif, cutlineDSName=cutline_shp, cropToCutline=False, format='GTiff')


# Parameterized function calls
def main():
    merged_output_file = f'{mask_dir}merged_esa_WORLDCOVER_ALGERIA.tif'
    reprojected_output_file = f'{mask_dir}reprojected_merged_clipped_esa_WORLDCOVER_ALGERIA.tif'
    resized_output_path = f'{mask_dir}mask_ALGERIA.tif'
    reprojected_mask_output = f'{mask_dir}open_street_map/mask_river/reprojected_mask.shp'
    input_shp = f'{mask_dir}open_street_map/mask_river/waterways1.shp'
    output_shp = f'{mask_dir}open_street_map/mask_river/waterways_mask1.shp'
    clipped_output_tif = f'{mask_dir}mask_river_ALGERIA.tif'
    esa_world_cover_r1 = f'{mask_dir}esa_world_cover_r1.tif'
    clip_mask_r = f'{mask_dir}reprojected_mask.shp'

    
    if not os.path.exists(f'{mask_dir}open_street_map/mask_river/'):
        os.makedirs(f'{mask_dir}open_street_map/mask_river/')
    # Set the paths to input and output files
    resized_output_path = f'{mask_dir}mask_ALGERIA.tif'
    

    # Open the input raster and get its projection
    input_ds = gdal.Open(esa_worldcover_1)
    input_proj = input_ds.GetProjection()

    # Open the cutline shapefile and get its layer
    cutline_ds = ogr.Open(clip_mask)
    cutline_layer = cutline_ds.GetLayer()

    # Get the bounding box of the cutline layer
    minX, maxX, minY, maxY = cutline_layer.GetExtent()

    # Set the output Spatial Reference System (SRS)
    output_srs = gdal.osr.SpatialReference()
    output_srs.ImportFromEPSG(32631)

    # Perform the warp operation with automatically determined outputBounds
    gdal.Warp(resized_output_path, input_ds, format='GTiff', outputBounds=[minX, minY, maxX, maxY],
              xRes=10.0, yRes=10.0, targetAlignedPixels=True, cutlineDSName=clip_mask,
              cutlineLayer='mitidja_aoi', cropToCutline=True, dstNodata=-9999.0,
              srcSRS='EPSG:4326', dstSRS=output_srs.ExportToWkt(),
              warpOptions=['INIT_DEST=-9999'])

    # Close the datasets
    input_ds = None
    cutline_ds = None

    # Run Python script to generate a buffer
    create_buffered_polygons(line_file, input_shp, buffer_width, 32631)

    reprojected_mask_output = f'{clip_mask}'
    # Clip input.shp using reprojected mask using OGR
    clip_input_using_mask(input_shp, reprojected_mask_output, output_shp)

    # Set source and target Spatial Reference Systems (SRS)
    source_srs = 'EPSG:32631'
    target_srs = 'EPSG:32631'

    # Perform reprojection and cutline operation
    gdal.Warp(clipped_output_tif, resized_output_path, format='GTiff', srcSRS=source_srs, dstSRS=target_srs,
              cutlineDSName=output_shp, cutlineLayer='waterways_mask1', cropToCutline=True)

    # Clip using gdalwarp
    #clip_using_gdalwarp(resized_output_path, clipped_output_tif, output_shp)
    print("mask creation completed.")


if __name__ == "__main__":
    main()
