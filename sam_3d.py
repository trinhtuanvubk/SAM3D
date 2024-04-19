import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import gc
gc.collect()
torch.cuda.empty_cache()

from utils import *

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)


mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)


#Reading the point cloud with laspy
pcd = laspy.read(os.path.join(data_path, "34FN2_18.las"))

#Transforming the point cloud to Numpy
pcd_np = np.vstack((pcd.x, pcd.y, pcd.z, (pcd.red/65535*255).astype(int), (pcd.green/65535*255).astype(int), (pcd.blue/65535*255).astype(int))).transpose()

#Ortho-Projection
orthoimage = cloud_to_image(pcd_np, 1.5)

#Plotting and exporting
fig = plt.figure(figsize=(np.shape(orthoimage)[1]/72, np.shape(orthoimage)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(orthoimage)
plt.axis('off')
plt.savefig("34FN2_18_orthoimage.jpg")



#Loading the las file from the disk
las = laspy.read(os.path.join(data_path,"ITC_BUILDING.las"))

#Transforming to a numpy array
coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

#Gathering the colors
r=(las.red/65535*255).astype(int)
g=(las.green/65535*255).astype(int)
b=(las.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()


resolution = 500

#Defining the position in the point cloud to generate a panorama
center_coordinates = [189, 60, 2]


#Function Execution
spherical_image, mapping = generate_spherical_image(center_coordinates, point_cloud, colors, resolution)

#Plotting with matplotlib
fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72, np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
plt.axis('off')

#Saving to the disk
plt.savefig("ITC_BUILDING_spherical_projection.jpg")



import time
temp_img = cv2.imread("ITC_BUILDING_spherical_projection.jpg")
image_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)

t0 = time.time()
result = mask_generator.generate(image_rgb)
t1 = time.time()


fig = plt.figure(figsize=(np.shape(image_rgb)[1]/72, np.shape(image_rgb)[0]/72))
fig.add_axes([0,0,1,1])

plt.imshow(image_rgb)
color_mask = sam_masks(result)
plt.axis('off')
plt.savefig("ITC_BUILDING_spherical_projection_segmented.jpg")




modified_point_cloud = color_point_cloud("ITC_BUILDING_spherical_projection_segmented.jpg", point_cloud, mapping)


import open3d as o3d
#Loading the las file from the disk
las = laspy.read(os.path.join(data_path,"ITC_BUILDING.las"))

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
o3d.io.write_point_cloud("hihi.ply", pcd)