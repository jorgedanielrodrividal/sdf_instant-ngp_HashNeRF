import torch
from torch.utils.data import DataLoader
from helper_functions.prepare_sdf_data import prepare_sdf_data
from src.models import SDFNeRFNetwork
from src.encoding import HashGridSDF
from src.data.datasets import BunnySDFDataset
from src.training import train_sdf_model
from .utils.visualization import visualize_sdf_slice

def main():
    # Configuration
    config = {
        "mesh_path" : "/content/bunny_data/bunny/reconstruction/bun_zipper.ply",
        "num_points": 1000,  # Total number of random points to generate
        "batch_size": 1024,
        "num_epochs": 100,
        "learning_rate": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "checkpoint_dir": "checkpoints"
    }

    # Generate SDF data
    sdf_data = prepare_sdf_data(config["mesh_path"], config["num_points"])

    # Update config with SDF data
    config.update(sdf_data)

    # Initialize dataset & dataloader
    dataset = BunnySDFDataset(points=config["points_random"],
                              sdf_values=config["sdf_values"])

    dataloader = DataLoader(dataset = dataset,
                            batch_size = config["batch_size"],
                            shuffle = True)

    # Test the DataLoader
    #sample_batch = next(iter(dataloader))
    #print(f"Batch Coordinates Shape: {sample_batch['coords'].shape}")  # (batch_size, 3)
    #print(f"Batch SDF Values Shape: {sample_batch['sdf'].shape}")      # (batch_size,)

    #Initialize SDF Hashing
    hash_encoder = HashGridSDF()

    #Initialize model
    model = SDFNeRFNetwork(hash_encoder)

    #Train the model
    trained_model = train_sdf_model(
    model=model,
    train_loader=dataloader,
    num_epochs=config["num_epochs"],
    lr=config["learning_rate"],
    device=config["device"],
    checkpoint_dir=config["checkpoint_dir"]
    )

    #Visualize results
    visualize_sdf_slice(trained_model)


if __name__ == "__main__":
    main()
