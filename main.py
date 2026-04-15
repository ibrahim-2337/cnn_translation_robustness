import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models import SimpleCNN
from src.data_loader import get_dataloaders
from src.utils import set_seed, eval_accuracy_on_shift
from src.engine import train_one_epoch

def run_experiment():
    cfg = {
        "data_root": "./data",
        "batch_size": 128,
        "epochs": 40,
        "lr": 0.01,
        "aug_pxs": [0, 2, 4],
        "downsampling_methods": ["maxpool", "avgpool"],
        "max_shift_px": 8,
        "seed": 31
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    shifts = list(range(0, cfg["max_shift_px"] + 1))

    for aug_px in cfg["aug_pxs"]:
        train_loader, test_loader = get_dataloaders(cfg["data_root"], cfg["batch_size"], aug_px)
        
        for ds in cfg["downsampling_methods"]:
            print(f"\nTraining: Aug={aug_px}px | DS={ds}")
            set_seed(cfg["seed"])
            model = SimpleCNN(num_classes=10, downsampling=ds).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.5)

            for epoch in range(cfg["epochs"]):
                loss, acc = train_one_epoch(model, train_loader, optimizer, device)
                scheduler.step()
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"  Epoch {epoch+1:02d}/{cfg['epochs']} | Loss {loss:.4f} | Acc {acc:.4f}")

            curve = [eval_accuracy_on_shift(model, test_loader, s, device) for s in shifts]
            results[(aug_px, ds)] = {
                "curve": curve,
                "robustness": float(np.mean(curve))
            }
            print(f"  Done | Robustness Score: {results[(aug_px, ds)]['robustness']:.4f}")

    return results

if __name__ == "__main__":
    results = run_experiment()
    # You could add code here to save or plot results
