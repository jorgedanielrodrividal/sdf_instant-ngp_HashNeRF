import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import os

def train_sdf_model(model, train_loader, num_epochs=100, lr=1e-4, device='cuda', checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.1**(1/num_epochs))
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            coords = batch['coords'].to(device)
            sdf_gt = batch['sdf'].to(device)

            sdf_pred = model(coords).squeeze()
            grad = model.get_sdf_gradient(coords)
            grad_norm = torch.norm(grad, dim=-1)

            sdf_loss = F.mse_loss(sdf_pred, sdf_gt)
            eikonal_loss = F.mse_loss(grad_norm, torch.ones_like(grad_norm))
            loss = sdf_loss + 0.1 * eikonal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

        scheduler.step()

    return model
