import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def shift_batch_right(x, shift_px):
    if shift_px == 0:
        return x
    b, c, h, w = x.shape
    out = torch.zeros_like(x)
    if shift_px < w:
        out[:, :, :, shift_px:] = x[:, :, :, :w - shift_px]
    return out

@torch.no_grad()
def eval_accuracy_on_shift(model, loader, shift_px, device):
    model.eval()
    total_correct = 0
    total_seen = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_shift = shift_batch_right(x, shift_px)
        logits = model(x_shift)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += x.size(0)
    return total_correct / total_seen
