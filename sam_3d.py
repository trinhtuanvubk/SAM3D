import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import time
import open3d as o3d

import os
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import gc
gc.collect()
torch.cuda.empty_cache()

from utils import *


def create_mask_generator(
    sam_checkpoint = "sam_vit_b_01ec64.pth",
    model_type = "vit_b",
    device = "cpu"):

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if device!="cpu":
        sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)
    return mask_generator

def gen_ortho_image(data_path="SAM_FOR_3D/DATA/34FN2_18.las",
                   output_path="images/34FN2_18_orthoimage.jpg"):
    #Reading the point cloud with laspy
    pcd = laspy.read(data_path)

    #Transforming the point cloud to Numpy
    pcd_np = np.vstack((pcd.x, pcd.y, pcd.z, 
                        (pcd.red/65535*255).astype(int),
                        (pcd.green/65535*255).astype(int), 
                        (pcd.blue/65535*255).astype(int))).transpose()

    #Ortho-Projection
    orthoimage = cloud_to_image(pcd_np, 1.5)

    #Plotting and exporting
    fig = plt.figure(figsize=(np.shape(orthoimage)[1]/72, np.shape(orthoimage)[0]/72))
    fig.add_axes([0,0,1,1])
    plt.imshow(orthoimage)
    plt.axis('off')
    plt.savefig(output_path)

def gen_spherical_image(data_path="SAM_FOR_3D/DATA/ITC_BUILDING.las",
                        output_path="images/ITC_BUILDING_spherical_projection.jpg",
                        resolution=500,
                        center_coordinates=[189,60,2]):
    """
    __DOCS__
    center_coordinates: Defining the position in the point cloud to generate a panorama
    """

    #Loading the las file from the disk
    las = laspy.read(data_path)

    #Transforming to a numpy array
    coords = np.vstack((las.x, las.y, las.z))
    point_cloud = coords.transpose()

    #Gathering the colors
    r=(las.red/65535*255).astype(int)
    g=(las.green/65535*255).astype(int)
    b=(las.blue/65535*255).astype(int)
    colors = np.vstack((r,g,b)).transpose()

    #Function Execution
    spherical_image, mapping = generate_spherical_image(center_coordinates, point_cloud, colors, resolution)
    print(spherical_image.shape)
    #Plotting with matplotlib
    fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72, np.shape(spherical_image)[0]/72))
    fig.add_axes([0,0,1,1])
    plt.imshow(spherical_image)
    plt.axis('off')

    #Saving to the disk
    plt.savefig(output_path)


def gen_spherical_masks(data_path="images/ITC_BUILDING_spherical_projection.jpg",
                        output_path="images/ITC_BUILDING_spherical_projection_segmented.jpg"):
    mask_generator = create_mask_generator()
    temp_img = cv2.imread(data_path)
    image_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

    t0 = time.time()
    result = mask_generator.generate(image_rgb)
    t1 = time.time()


    fig = plt.figure(figsize=(np.shape(image_rgb)[1]/72, np.shape(image_rgb)[0]/72))
    fig.add_axes([0,0,1,1])

    plt.imshow(image_rgb)
    # color_mask = sam_masks(result)
    plt.axis('off')
    plt.savefig(output_path)


def pipeline(data_path="SAM_FOR_3D/ITC_BUILDING.las",
                        output_path="images/hihi.ply",
                        resolution=500,
                        center_coordinates=[189,60,2]):
    
    """
    __DOCS__
    center_coordinates: Defining the position in the point cloud to generate a panorama
    """

    #Loading the las file from the disk
    las = laspy.read(data_path)

    #Transforming to a numpy array
    coords = np.vstack((las.x, las.y, las.z))
    point_cloud = coords.transpose()

    #Gathering the colors
    r=(las.red/65535*255).astype(int)
    g=(las.green/65535*255).astype(int)
    b=(las.blue/65535*255).astype(int)
    colors = np.vstack((r,g,b)).transpose()

    #Function Execution
    spherical_image, mapping = generate_spherical_image(center_coordinates, point_cloud, colors, resolution)


    modified_point_cloud = color_point_cloud_v2(spherical_image , point_cloud, mapping)
    las = export_point_cloud("pcd_results.las", modified_point_cloud)

    # #Loading the las file from the disk
    # las = laspy.read(os.path.join(data_path,"ITC_BUILDING.las"))

    #Transforming to a numpy array
    coords = np.vstack((las.x, las.y, las.z))
    point_cloud = coords.transpose()


    #Gathering the colors
    r=(las.red/65535*255).astype(int)
    g=(las.green/65535*255).astype(int)
    b=(las.blue/65535*255).astype(int)
    colors = np.vstack((r,g,b)).transpose()


    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Save to PLY
    o3d.io.write_point_cloud(output_path, pcd)



if __name__=="__main__":
    gen_spherical_image()