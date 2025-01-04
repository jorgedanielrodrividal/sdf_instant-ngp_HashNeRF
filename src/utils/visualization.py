import torch
import matplotlib.pyplot as plt

def visualize_sdf_slice(model, resolution=100, z_value=0.0, device='cuda'):
    model.eval()
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    xx, yy = torch.meshgrid(x, y)
    points = torch.stack([xx.flatten(), yy.flatten(), torch.ones_like(xx.flatten()) * z_value], dim=-1).to(device)

    with torch.no_grad():
        sdf_values = model(points).cpu().numpy().reshape(resolution, resolution)

    plt.figure(figsize=(10, 10))
    plt.imshow(sdf_values, extent=[-1, 1, -1, 1], cmap='RdBu')
    plt.colorbar(label='SDF Value')
    plt.title(f'SDF Slice at z={z_value}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
