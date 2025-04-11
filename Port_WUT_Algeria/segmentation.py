# -*- coding: utf-8 -*-
"""
WaterSat
author: Tim Martijn Hessels

"""

import sys
import json
import glob
import re
from osgeo import gdal
import numpy as np
import glymur
import os
import heapq
from math import sqrt
import subprocess
import watertools.General.data_conversions as DC

from config import (
    s2_tile,
    output_dir,
    number_of_segments,
    output_sen2_directory,
    clip_mask,
)


def main():

    base_path = f'{output_dir}masked/'

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

    print(
        f"Top-left pixel of TIF file in JP2 file: (X: {corresponding_pixel['jp2_pixel_x']}, Y: {corresponding_pixel['jp2_pixel_y']})")

    x_pixels = intersection_info["intersection_width"]
    y_pixels = intersection_info["intersection_height"]

    print(f'x_pixels: {x_pixels}, y_pixels: {y_pixels}')

    split_x = split_y = 1

    # SNIC parameters
    compactness = 0.001
    home_folder = base_path

    s2_raw_folder = os.path.join(home_folder, s2_tile, "RAW_S2")

    for y_tile in range(1, split_y + 1):
        for x_tile in range(1, split_x + 1):

            Array_number = 1

            Total_pixels = x_pixels * y_pixels
            file_snic_out = "SNIC_%s_V2.tif" % (number_of_segments)
            x_id_start = 0
            x_id_end = x_pixels
            y_id_start = 0
            y_id_end = y_pixels
            tile_str = ""

            Delta_x_end = x_id_end - x_id_start
            Delta_y_end = y_id_end - y_id_start

            number_of_segments_part = int(np.ceil(number_of_segments * (Delta_x_end * Delta_y_end) / Total_pixels))

            folder_out = os.path.join(home_folder, s2_tile, "Segmentation_Tiffs")
            if not os.path.exists(folder_out):
                os.makedirs(folder_out)

            print("base_path C :",base_path)
            
            date_pattern = r"\d{8}"  # Match 8 consecutive digits (date format YYYYMMDD)

            directories = [folder for folder in os.listdir(base_path) if re.match(date_pattern, folder)]
            print("directories C :",directories)

            band_files = [
                os.path.join(base_path, folder, f"band{i}.tif") 
                for folder in directories 
                for i in [2, 3, 4, 8, 5]  # Only specified bands
            ]

            dest_list = [gdal.Open(file_path) for file_path in band_files]
            bands = [dest.GetRasterBand(1).ReadAsArray() for dest in dest_list]
            #print("bands C :",bands[2])
            B3 = bands[2]
            proj = dest_list[0].GetProjection()
            geo = dest_list[0].GetGeoTransform()

            B = np.array(bands)
            #print("calculate C")
            C = np.einsum('xyz->yzx', B)
            #print("calculate A")
            A = C.tolist()

            # load image
            number_of_pixels = B3.shape[0] * B3.shape[1]
            #print("number_of_pixels C :",number_of_pixels)
           
            segmentation, _, number_of_segments2 = snic(
                A, number_of_segments_part, compactness,
                update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

            # import matplotlib.pyplot as plt
            # from skimage.segmentation import mark_boundaries

            # show the output of SNIC
            # fig = plt.figure("SNIC with %d segments" % number_of_segments)
            # plt.imshow(mark_boundaries(B3, np.array(segmentation)))
            # plt.show()
            # D = mark_boundaries(B3, np.array(segmentation))
            snic_array = np.array(segmentation)
            output_SNIC = os.path.join(home_folder, s2_tile, file_snic_out)
            DC.Save_as_tiff(output_SNIC, snic_array, geo, proj)

    return ()


def snic(
        image,
        seeds,
        compactness,
        nd_computation=None,
        image_distance=None,
        update_func=None):
    class Queue(object):
        def __init__(self, _buffer_size=0):
            self.heap = []
            self._sub_idx = 0

        def add(self, priority, value):
            heapq.heappush(self.heap, (priority, self._sub_idx, value))
            self._sub_idx += 1

        def is_empty(self):
            return len(self.heap) == 0

        def pop_value(self):
            return heapq.heappop(self.heap)[2]

        def pop(self):
            return heapq.heappop(self.heap)

        def length(self):
            return len(self.heap)

    class NdComputations(object):

        def __init__(self, lerp, nd_distance):
            self.lerp = lerp
            self.nd_distance = nd_distance

    image_size = [len(image), len(image[0])]
    label_map = [[-1] * image_size[1] for _ in range(image_size[0])]
    distance_map = [[sys.float_info.max] * image_size[1] for _ in range(image_size[0])]

    if nd_computation is None:
        nd_computations = {
            "1": NdComputations(lerp1, norm1_sqr_arr),
            "2": NdComputations(lerp2, norm2_sqr_arr),
            "3": NdComputations(lerp3, norm3_sqr_arr),
            "nd": NdComputations(lerp_nd, norm_nd_sqr_arr),
        }
        nd_computation = nd_computations["nd"]
    lerp = nd_computation.lerp

    if type(seeds) is int:
        # generate grid and flatten it into a list
        seeds = [seed for row in compute_grid(image_size, seeds) for seed in row]

        number_of_superpixels = len(seeds)
    else:
        # assume seeds is an iterable
        number_of_superpixels = len(seeds)

    if image_distance is None:
        image_distance = create_augmented_snic_distance(image_size, number_of_superpixels, compactness)

    # create centroids
    centroids = [[pos, image[pos[1]][pos[0]], 0] for pos in seeds]  # [position, avg color, #pixels]

    # create priority queue
    queue = Queue(image_size[0] * image_size[1] * 4)  # [position, color, centroid_idx]
    q_add = queue.add  # cache some functions
    q_pop = queue.pop
    # we create a priority queue and fill with the centroids itself. Since the python priority queue can not
    # handle multiple entries with the same key, we start inserting the super pixel seeds with negative values. This
    # makes sure they get processed before any other pixels. Since distances can not be negative, all new
    # pixels will have a positive value, and therefore will be handled only after all seeds have been processed.
    for k in range(number_of_superpixels):
        init_centroid = centroids[k]

        q_len = -queue.length()
        q_add(q_len, [init_centroid[0], init_centroid[1], k])
        distance_map[init_centroid[0][1]][init_centroid[0][0]] = q_len

    # classification
    classified_pixels = 0
    # while not q_empty(): -> replaced with "try: while True:" to speed-up code (~1sec with 50k iterations)
    try:
        while True:
            # get pixel that has the currently smallest distance to a centroid
            item = q_pop()
            candidate_distance = item[0]
            candidate = item[2]
            candidate_pos = candidate[0]

            # test if pixel is not already labeled
            # if label_map[candidate_pos[1] * im_width + candidate_pos[0]] == -1:
            if label_map[candidate_pos[1]][candidate_pos[0]] == -1:
                centroid_idx = candidate[2]

                # label new pixel
                label_map[candidate_pos[1]][candidate_pos[0]] = centroid_idx
                #
                distance_map[candidate_pos[1]][candidate_pos[0]] = candidate_distance
                # label_map[candidate_pos[1] * im_width + candidate_pos[0]] = centroid_idx
                classified_pixels += 1

                # online update of centroid
                centroid = centroids[centroid_idx]
                num_pixels = centroid[2] + 1
                lerp_ratio = 1 / num_pixels

                # adjust centroid position
                centroid[0] = lerp2(centroid[0], candidate_pos, lerp_ratio)
                # update centroid color
                centroid[1] = lerp(centroid[1], candidate[1], lerp_ratio)
                # adjust number of pixels counted towards this super pixel
                centroid[2] = num_pixels

                # add new candidates to queue
                neighbours, neighbour_num = get_4_neighbourhood_1(candidate_pos, image_size)
                for i in range(neighbour_num):
                    neighbour_pos = neighbours[i]
                    # Check if neighbour is already labeled, as these pixels would get discarded later on.
                    # We filter them here as queue insertions are expensive
                    npx = neighbour_pos[0]
                    npy = neighbour_pos[1]
                    # if label_map[neighbour_pos[1] * im_width + neighbour_pos[0]] == -1:
                    if label_map[npy][npx] == -1:
                        neighbour_color = image[npy][npx]
                        neighbour = [neighbour_pos, neighbour_color, centroid_idx]

                        distance = image_distance(neighbour_pos, centroid[0], neighbour_color, centroid[1])

                        # test if another candidate with a lower distance, is not already
                        # registered to this pixel
                        if distance_map[npy][npx] >= distance:
                            distance_map[npy][npx] = distance
                            q_add(distance, neighbour)

                # status update
                if (update_func is not None) and (classified_pixels % 10000 == 0):
                    update_func(classified_pixels)
    except IndexError:
        pass

    # do a 100% status update (if we haven't done it already in the last iteration)
    if (update_func is not None) and (classified_pixels % 10000 != 0):
        update_func(classified_pixels)

    return label_map, distance_map, centroids


def lerp1(a, b, w):
    return (a * (1 - w)) + (b * w)


def lerp2(a, b, w):
    return [
        (a[0] * (1 - w)) + (b[0] * w),
        (a[1] * (1 - w)) + (b[1] * w)]


def lerp3(a, b, w):
    return [
        (a[0] * (1 - w)) + (b[0] * w),
        (a[1] * (1 - w)) + (b[1] * w),
        (a[2] * (1 - w)) + (b[2] * w)]


def lerp_nd(a, b, w):
    u = 1 - w

    def lerp1_w(x, y):
        return (x * u) + (y * w)

    return list(map(lerp1_w, a, b))


def get_4_neighbourhood_1(pos, image_size):
    # outputs candidates 1 pixel away from the image border.
    # this way we can use interp2d_lin_unsafe instead and safe costly boundary checks
    n = 0
    neighbourhood = [None, None, None, None]

    x = pos[0]
    y = pos[1]

    if x - 1 >= 0:
        neighbourhood[0] = [x - 1, y]
        n += 1

    if y - 1 >= 0:
        neighbourhood[n] = [x, y - 1]
        n += 1

    if x + 1 < image_size[1]:
        neighbourhood[n] = [x + 1, y]
        n += 1

    if y + 1 < image_size[0]:
        neighbourhood[n] = [x, y + 1]
        n += 1

    return neighbourhood, n


def norm2_sqr_arr(a, b):
    x = a[0] - b[0]
    return x * x


def norm2_sqr(x, y):
    """
    Squared 2-norm for 2d vectors
    :param x:
    :param y:
    :return:
    """
    return (x * x) + (y * y)


def norm1_sqr_arr(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return (x * x) + (y * y)


def norm3_sqr(x, y, z):
    """
    Squared 2-norm for 3d vectors
    :param x:
    :param y:
    :param z:
    :return:
    """
    return (x * x) + (y * y) + (z * z)


def norm3_sqr_arr(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    z = a[2] - b[2]
    return (x * x) + (y * y) + (z * z)


def norm_nd_sqr_arr(a, b):
    def sub_sqr(x, y):
        d = x - y
        return d * d

    return sum(map(sub_sqr, a, b))


def compute_grid(image_size, number_of_pixels):
    """

    :param image_size: [row, cols]
    :param number_of_pixels:
    :return:
    """
    image_size_y = float(image_size[0])
    image_size_x = float(image_size[1])

    # compute grid size
    image_ratio = image_size_x / image_size_y
    num_sqr = sqrt(number_of_pixels)

    grid_size = [int(max(1.0, num_sqr * image_ratio) + 1), int(max(1.0, num_sqr / image_ratio) + 1)]

    # create grid
    full_step = [image_size_x / float(grid_size[0]), image_size_y / float(grid_size[1])]
    half_step = [full_step[0] / 2.0, full_step[1] / 2.0]
    grid = [[[
        int(half_step[0] + x * full_step[0]),
        int(half_step[1] + y * full_step[1])
    ] for x in range(grid_size[0])] for y in range(grid_size[1])]
    return grid


def create_augmented_snic_distance(image_size, number_of_superpixels, compactness):
    # compute normalization factors
    si = 1 / sqrt((image_size[0] * image_size[1]) / number_of_superpixels)
    mi = 1 / float(compactness)

    def snic_distance_augmented(pa, pb, ca, cb, ss=si, mm=mi):
        return snic_distance_mod(pa, pb, ca, cb, ss, mm)

    return snic_distance_augmented


def snic_distance_mod(pos_i, pos_j, col_i, col_j, si, mi):
    pos_d = norm2_sqr(pos_j[0] - pos_i[0], pos_j[1] - pos_i[1]) * si
    col_d = norm_nd_sqr_arr(col_i, col_j) * mi
    distance = pos_d + col_d
    return distance


def Get_Proj(filename_mtl):  
    
    meta = open(filename_mtl, "r")
    for line in meta:
        if re.match("(.*)HORIZONTAL_CS_CODE(.*)", line): # search in metadata for Projection
           proj = int(line.split("EPSG:")[1].split("<")[0])
                     
    return(proj)    
    

def Get_S2_MTL_folder(folder):
    
    folder_GRANULE = os.path.join(folder, "GRANULE")
    os.chdir(folder_GRANULE)
    filename_S2 = glob.glob("L2*")[0]     
    folder_GRANULE_S2 = os.path.join(folder_GRANULE, filename_S2)    
    
    return(folder_GRANULE_S2)

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



if __name__ == "__main__":
    main()

