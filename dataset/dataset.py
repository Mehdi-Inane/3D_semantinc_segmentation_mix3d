from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union
from utils.ply_utils import read_ply
import numpy as np
import os
import albumentations as A
from .transforms import transforms as V
from .transforms.composition import Compose
import random
import scipy



def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
  """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [
        np.linspace(d_min, d_max, d)
        for d_min, d_max, d in zip(
            coords_min - granularity,
            coords_min + granularity * (noise_dim - 2),
            noise_dim,
        )
    ]
    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0
    )
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud






# Take areas 1,2,3,6 for training, 4 for validation and 5 for test

class S3DISDataset(Dataset):

  def __init__(self,data_dir,
               mode = 'train',
               color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
               color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
               flip_in_center = True,
               volume_augmentation = None,
               image_augmentation = None):
            self.data_dir = data_dir
            self.mode = mode
            files = os.listdir(self.data_dir)
            # Take the appropriate areas according to train/val/test
            if mode == 'train':
              vals = ['Area_1','Area_2','Area_3','Area_6']
            elif mode == 'val':
              vals = ['Area_4']
            else:
              vals = ['Area_5']
            self.data = [os.path.join(self.data_dir,f) for f in files if f[:6] in vals]
            self.flip_center = flip_in_center
            self.volume_augmentation = self.get_transform_volument()
            self.image_augmentation = self.get_albumentation_transform()
            self.normalize_color = A.Normalize(mean=color_mean,std = color_std)



  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    filename = self.data[idx]
    all_data = read_ply(filename)
    points = np.stack([np.array(all_data['x']),np.array(all_data['y']),np.array(all_data['z'])],axis=1)
    color = np.stack([all_data['red'],all_data['green'],all_data['blue']],axis = 1)
    #normals = np.stack([all_data['n_x'],all_data['n_y'],all_data['n_z']],axis = 1)
    labels = all_data['label']

    if self.mode == 'train':
      # Centering points
      points -= points.mean(0)
      # Adding noise
      points += np.random.uniform(points.min(0), points.max(0)) / 2

      if self.flip_center:
        points = flip_in_center(points)

      for i in (0, 1):
        if random.random() < 0.5:
          coord_max = np.max(points[:, i])
          points[:, i] = coord_max - points[:, i]
        if random.random() < 0.95:
          for granularity, magnitude in ((0.2, 0.4), (0.8, 1.6)):
                coordinates = elastic_distortion(points, granularity, magnitude)
        aug = self.volume_augmentation(
                points=points, features=color, labels=labels,
            )
        points, color, labels = (
                aug["points"],
                aug["features"],
                aug["labels"],
            )
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.image_augmentation(image=pseudo_image)["image"])
        pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
        color = np.squeeze(self.normalize_color(image=pseudo_image)["image"])
        features = color
        return points, features,labels



  def get_transform_volument(self):
    return Compose([
    V.Scale3d(always_apply=True, p=0.5, scale_limit=([-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1])),
    V.RotateAroundAxis3d(always_apply=True, p=0.5, axis=[0, 0, 1], rotation_limit=(-3.141592653589793, 3.141592653589793)),
    V.RotateAroundAxis3d(always_apply=True, p=0.5, axis=[0, 1, 0], rotation_limit=(-0.13089969389957, 0.13089969389957)),
    V.RotateAroundAxis3d(always_apply=True, p=0.5, axis=[1, 0, 0], rotation_limit=(-0.13089969389957, 0.13089969389957))
], p=1.0)

  def get_albumentation_transform(self):
    return A.Compose([
A.RandomBrightnessContrast(always_apply=True, brightness_by_max=True, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
A.RGBShift(always_apply=True, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20), p=0.5)
], p=1.0)


def flip_in_center(coordinates):
    # moving coordinates to center
    coordinates -= coordinates.mean(0)
    aug = Compose(
        [
            V.Flip3d(axis=(0, 1, 0), always_apply=True),
            V.Flip3d(axis=(1, 0, 0), always_apply=True),
        ]
    )
    first_crop = coordinates[:, 0] > 0
    first_crop &= coordinates[:, 1] > 0
    # x -y
    second_crop = coordinates[:, 0] > 0
    second_crop &= coordinates[:, 1] < 0
    # -x y
    third_crop = coordinates[:, 0] < 0
    third_crop &= coordinates[:, 1] > 0
    # -x -y
    fourth_crop = coordinates[:, 0] < 0
    fourth_crop &= coordinates[:, 1] < 0

    if first_crop.size > 1:
        coordinates[first_crop] = aug(points=coordinates[first_crop])["points"]
    if second_crop.size > 1:
        minimum = coordinates[second_crop].min(0)
        minimum[2] = 0
        minimum[0] = 0
        coordinates[second_crop] = aug(points=coordinates[second_crop])["points"]
        coordinates[second_crop] += minimum
    if third_crop.size > 1:
        minimum = coordinates[third_crop].min(0)
        minimum[2] = 0
        minimum[1] = 0
        coordinates[third_crop] = aug(points=coordinates[third_crop])["points"]
        coordinates[third_crop] += minimum
    if fourth_crop.size > 1:
        minimum = coordinates[fourth_crop].min(0)
        minimum[2] = 0
        coordinates[fourth_crop] = aug(points=coordinates[fourth_crop])["points"]
        coordinates[fourth_crop] += minimum

    return coordinates