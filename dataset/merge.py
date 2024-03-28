import numpy as np
import torch
from random import random

class SimpleCollateMergeToTensor:
    def __init__(self, device='cpu', ignore_label=255, scenes=2, downsample_to=None, place_nearby=False, place_far=False, proba=1):
        self.device = device
        self.scenes = scenes
        self.ignore_label = ignore_label
        self.downsample_to = downsample_to
        self.place_nearby = place_nearby
        self.place_far = place_far
        self.proba = proba

    def downsample(self, coords, feats, labels):
        """Randomly downsamples a point cloud to a fixed number of points."""
        if self.downsample_to is not None and coords.shape[0] > self.downsample_to:
            indices = np.random.choice(coords.shape[0], self.downsample_to, replace=False)
            coords = coords[indices]
            feats = feats[indices]
            labels = labels[indices]
        return coords, feats, labels

    def __call__(self, batch):
        new_batch = []
        for i in range(0, len(batch), self.scenes):
            batch_coordinates = []
            batch_features = []
            batch_labels = []

            for j in range(min(len(batch[i:]), self.scenes)):
                coords, feats, labels = batch[i + j]
                # Downsample each point cloud
                coords_downsampled, feats_downsampled, labels_downsampled = self.downsample(coords, feats, labels)
                
                # Convert to tensors and move to the specified device
                coords_t = torch.from_numpy(coords_downsampled).float().to(self.device)
                feats_t = torch.from_numpy(feats_downsampled).float().to(self.device)
                labels_t = torch.from_numpy(labels_downsampled).long().to(self.device)

                batch_coordinates.append(coords_t)
                batch_features.append(feats_t)
                batch_labels.append(labels_t)

            # Handling the placement of point clouds
            if (len(batch_coordinates) == 2) and self.place_nearby:
                border = batch_coordinates[0][:, 0].max()
                batch_coordinates[1][:, 0] += border
            elif (len(batch_coordinates) == 2) and self.place_far:
                displacement = torch.from_numpy(np.random.uniform(-2000, 2000, size=batch_coordinates[1].shape)).float().to(self.device)
                batch_coordinates[1] += displacement

            new_batch.append(
                (
                    torch.cat(batch_coordinates, 0),
                    torch.cat(batch_features, 0),
                    torch.cat(batch_labels, 0),
                )
            )

        return new_batch
