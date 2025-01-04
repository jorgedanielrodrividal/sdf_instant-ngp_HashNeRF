import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
from src.encoding import HashGridSDF, VanillaHashGrid

#Implementation of HashNeRF, see: https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/run_nerf_helpers.py#L77
class SDFNeRFNetwork(nn.Module):
  def __init__(self, hash_encoder: HashGridSDF, D=8, W=64, skips=[4]):
    super().__init__()
    self.hash_encoder = hash_encoder
    self.D = D # Number of layers in the MLP
    self.W = W # Number of neurons in each layer
    self.input_ch = hash_encoder.n_features # Input features from HashGridSDF
    self.skips = skips

    #Define MLP block with skip connections
    self.layers = nn.ModuleList()
    #print(f"Length of input_ch:\n {self.input_ch}\n")
    self.layers.append(nn.Linear(self.input_ch, self.W)) # First layer
    for i in range(1, self.D):
      if i in self.skips:
        self.layers.append(nn.Linear(self.W + self.input_ch, self.W))  # Skip connection
      else:
        self.layers.append(nn.Linear(self.W, self.W)) # Regular layer

    #Output layer for SDF
    self.output_layer = nn.Linear(self.W, 1)

  def forward(self, x):
    #print("Input to forward:", x.shape)
    features = self.hash_encoder(x) # Encode input 3D points
    #print("Features shape:", features.shape)

    h = features
    #print(f"features:\n {h.shape}")
    for i, layer in enumerate(self.layers):
      #h = F.relu(layer(h)) # Apply ReLU activation
      if i in self.skips:
        h = torch.cat([features, h], dim=-1) # Skip connection
        #print(f"After skip connection at Layer {i}, shape:", h.shape)
      h = F.relu(layer(h)) # Apply ReLU activation
      #print(f"Layer {i} output shape:", h.shape)
    #One SDF value per input point
    sdf = self.output_layer(h) # Predict SDF value; SDF has shape (batch_size, 1)
    #print(f"SDF tensor is:\n {sdf.shape}\n")
    return sdf #return print() evaluates to None. #print(f"SDF tensor is:\n {sdf.shape}\n")

  def get_sdf_gradient(self, x, epsilon=1e-4):
    """
    Compute the gradient of the SDF with respect to the input coordinates.
    Args:
        x: Input tensor of shape (batch_size, 3) representing 3D coordinates.
    Returns:
        grad: Gradient tensor of shape (batch_size, 3).
    """
    with torch.enable_grad():
      # # Ensure `x` is on the same device as the model
      # x = x.to(next(self.parameters()).device)

      x.requires_grad_(True) # Enable gradient computation for the input
      #print(f"Input before forward pass:\n {x.shape}")
      sdf = self.forward(x) # Compute SDF for the input
      #print(sdf)
      grad = torch.autograd.grad(
          outputs=sdf.sum(),  # Sum is used to create a scalar loss
          inputs=x,
          create_graph=True, # Retain the computational graph for higher-order gradients
          retain_graph=True
      )[0] # The gradient of sdf.sum() with respect to x
    return grad
