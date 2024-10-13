import torch
import torch.nn as nn
from tqdm import tqdm
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from KAN import KAN

def test_mul():
    kan = KAN([2, 2, 1], base_activation=nn.Identity)
    optimizer = torch.optim.LBFGS(kan.parameters(), lr=1)

    # Move model to CPU to avoid potential GPU issues
    kan = kan.cpu()

    with tqdm(range(100)) as pbar:
        for i in pbar:
            loss, reg_loss = None, None

            def closure():
                optimizer.zero_grad()
                # Generate input data and move to CPU
                x = torch.rand(1024, 2).cpu()
                y = kan(x, update_grid=(i % 20 == 0))

                # Check dimensions
                print(f"x shape: {x.shape}, y shape: {y.shape}")
                assert y.shape == (1024, 1)

                nonlocal loss, reg_loss
                u = x[:, 0]
                v = x[:, 1]
                loss = nn.functional.mse_loss(y.squeeze(-1), (u + v) / (1 + u * v))
                reg_loss = kan.regularization_loss(1, 0)

                # Check for NaNs in loss values
                if torch.isnan(loss) or torch.isnan(reg_loss):
                    print("NaN detected in loss calculations. Skipping step.")
                    return torch.tensor(0.0)

                (loss + 1e-5 * reg_loss).backward()
                return loss + reg_loss

            # Use anomaly detection to trace the source of the error
            with torch.autograd.set_detect_anomaly(True):
                optimizer.step(closure)

            # Update progress bar with loss values
            pbar.set_postfix(mse_loss=loss.item() if loss is not None else 0, reg_loss=reg_loss.item() if reg_loss is not None else 0)

    # Print spline weights for each layer
    for layer in kan.layers:
        print(layer.spline_weight)

def main():
    test_mul()

if __name__ == "__main__":
    main()