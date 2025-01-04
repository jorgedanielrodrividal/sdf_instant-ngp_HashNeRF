import torch
from torch.utils.data import Dataset

class BunnySDFDataset(Dataset):
    def __init__(self, points, sdf_values):
        """
        Initialize the dataset with sampled points and their SDF values.

        Args:
        points (numpy.ndarray): Array of 3D points (num_points, 3).
        sdf_values (numpy.ndarray): Array of SDF values corresponding to the points (num_points,).
        """
        self.points = torch.tensor(points, dtype=torch.float32)
        self.sdf_values = torch.tensor(sdf_values, dtype=torch.float32)

    def __len__(self):
        return len(self.points) #self.points

    def __getitem__(self, idx):
        """
        Retrieve the 3D point and its SDF value at the given index.

        Args:
        idx (int): Index of the sample.

        Returns:
        dict: Dictionary containing 'coords' (3D point) and 'sdf' (SDF value).
        """
        return {
            "coords": self.points[idx],
            "sdf": self.sdf_values[idx]
        }
