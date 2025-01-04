import torch
import torch.nn as nn
import numpy as np

class HashGridSDF(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.n_features = n_levels * n_features_per_level
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        # Initialize hash tables
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

        # Compute level resolutions
        self.level_resolutions = [
            int(np.floor(base_resolution * np.exp(
                level * np.log(finest_resolution / base_resolution) / (n_levels - 1)
            ))) for level in range(n_levels)
        ]

    def hash_coords(self, coords, resolution):
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        scaled_coords = coords * resolution
        coord_int = scaled_coords.long()
        hashed = (coord_int * primes).sum(dim=-1)
        return hashed % self.hash_tables[0].num_embeddings

    def forward(self, coords):
        # Ensure coords is the right shape (batch_size, 3)
        if coords.dim() == 3:
            coords = coords.squeeze(1)

        output_features = []
        for level, resolution in enumerate(self.level_resolutions):
            scaled_coords = coords * resolution
            coords_floor = torch.floor(scaled_coords)
            coords_frac = scaled_coords - coords_floor

            # Get all 8 corners of the cube
            corners = torch.stack([
                coords_floor + torch.tensor([i, j, k], device=coords.device)
                for i in [0, 1] for j in [0, 1] for k in [0, 1]
            ], dim=1)  # Shape: (batch_size, 8, 3)

            # Reshape corners for hashing
            corners_flat = corners.view(-1, 3)  # Shape: (batch_size * 8, 3)

            # Hash and get features for all corners
            hashed_indices = self.hash_coords(corners_flat, resolution)
            corner_features = self.hash_tables[level](hashed_indices)

            # Reshape corner features
            corner_features = corner_features.view(-1, 8, self.n_features_per_level)

            # Interpolate
            interpolated = self.trilinear_interpolate(corner_features, coords_frac)
            output_features.append(interpolated)

        return torch.cat(output_features, dim=-1)

    def trilinear_interpolate(self, features, coords_frac):
        # features: (batch_size, 8, feature_dim)
        # coords_frac: (batch_size, 3)

        x, y, z = coords_frac[..., 0], coords_frac[..., 1], coords_frac[..., 2]

        # Add singleton dimensions for broadcasting
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        z = z.unsqueeze(-1)

        c000 = features[:, 0]  # (0,0,0)
        c001 = features[:, 1]  # (0,0,1)
        c010 = features[:, 2]  # (0,1,0)
        c011 = features[:, 3]  # (0,1,1)
        c100 = features[:, 4]  # (1,0,0)
        c101 = features[:, 5]  # (1,0,1)
        c110 = features[:, 6]  # (1,1,0)
        c111 = features[:, 7]  # (1,1,1)

        # Interpolation along x
        c00 = c000 * (1 - x) + c100 * x
        c01 = c001 * (1 - x) + c101 * x
        c10 = c010 * (1 - x) + c110 * x
        c11 = c011 * (1 - x) + c111 * x

        # Interpolation along y
        c0 = c00 * (1 - y) + c10 * y
        c1 = c01 * (1 - y) + c11 * y

        # Interpolation along z
        c = c0 * (1 - z) + c1 * z

        return c
    
class VanillaHashGrid(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19, base_resolution=16, finest_resolution=512):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.n_features = n_levels * n_features_per_level
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        # Initialize hash tables
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2**log2_hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for table in self.hash_tables:
            nn.init.uniform_(table.weight, -1e-4, 1e-4)

        # Compute level resolutions
        self.level_resolutions = [
            int(np.floor(base_resolution * np.exp(
                level * np.log(finest_resolution / base_resolution) / (n_levels - 1)
            ))) for level in range(n_levels)
        ]

    def hash_coords(self, coords, resolution):
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        scaled_coords = coords * resolution
        coord_int = scaled_coords.long()
        hashed = (coord_int * primes).sum(dim=-1)
        return hashed % self.hash_tables[0].num_embeddings

    def forward(self, coords):
        # Ensure coords is the right shape (batch_size, 3)
        if coords.dim() == 3:
            coords = coords.squeeze(1)

        output_features = []
        for level, resolution in enumerate(self.level_resolutions):
            scaled_coords = coords * resolution
            coords_floor = torch.floor(scaled_coords)
            coords_frac = scaled_coords - coords_floor

            # Get all 8 corners of the cube
            corners = torch.stack([
                coords_floor + torch.tensor([i, j, k], device=coords.device)
                for i in [0, 1] for j in [0, 1] for k in [0, 1]
            ], dim=1)  # Shape: (batch_size, 8, 3)

            # Reshape corners for hashing
            corners_flat = corners.view(-1, 3)  # Shape: (batch_size * 8, 3)

            # Hash and get features for all corners
            hashed_indices = self.hash_coords(corners_flat, resolution)
            corner_features = self.hash_tables[level](hashed_indices)

            # Reshape corner features
            corner_features = corner_features.view(-1, 8, self.n_features_per_level)

            # Interpolate
            interpolated = self.trilinear_interpolate(corner_features, coords_frac)
            output_features.append(interpolated)

        return torch.cat(output_features, dim=-1)
