## COLAB_PLAIN34_SEED7.py
## ════════════════════════════════════════════════════════════
## Single run: Plain-34, seed=7, Protocol B (Cosine, 100ep)
## This is the ONE missing result from COLAB_MULTISEED.py
##
## Runtime: ~60 minutes on T4 GPU
## ════════════════════════════════════════════════════════════

## ── CELL 1: Mount Drive + setup ──────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys, os

ZIP_PATH    = "/content/drive/MyDrive/resnet34.zip"
PROJECT_DIR = "/content/resnet34-amazon-products"

if not os.path.exists(PROJECT_DIR):
    subprocess.run(["unzip", "-q", ZIP_PATH, "-d", "/content/"], check=True)
    print("✓ Unzipped")
else:
    print("✓ Already unzipped")

sys.path.insert(0, PROJECT_DIR)
subprocess.run(["pip", "install", "-e", PROJECT_DIR, "-q", "--no-deps"], check=True)
print("✓ Ready")

## ── CELL 2: Run plain34 seed=7 ───────────────────────────────
from src.models.model_factory import ModelFactory
import torch, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import numpy as np, json, time, random

SEED   = 7
EPOCHS = 100
LR     = 0.1
MEAN   = (0.4914, 0.4822, 0.4465)
STD    = (0.2470, 0.2435, 0.2616)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'none'}")

# Seed
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# Data
tr = T.Compose([T.RandomCrop(32, padding=4, padding_mode='reflect'),
                T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
vt = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
rng = np.random.default_rng(SEED); idx = rng.permutation(50000)

tr_ds = torchvision.datasets.CIFAR10('/tmp/c10', True,  tr, download=True)
va_ds = torchvision.datasets.CIFAR10('/tmp/c10', True,  vt, download=False)
te_ds = torchvision.datasets.CIFAR10('/tmp/c10', False, vt, download=False)

tr_ld = DataLoader(Subset(tr_ds, idx[:45000]), 128, shuffle=True,  num_workers=2, pin_memory=True)
va_ld = DataLoader(Subset(va_ds, idx[45000:]), 256, shuffle=False, num_workers=2, pin_memory=True)
te_ld = DataLoader(te_ds, 256, shuffle=False, num_workers=2, pin_memory=True)

# Model
model = ModelFactory.create("plain34", num_classes=10, dataset="cifar10").to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
assert n_params == 21_108_298, f"Param mismatch: {n_params:,}"
print(f"✓ Plain-34: {n_params:,} params")

# Optimizer
decay    = [p for _, p in model.named_parameters() if p.ndim >= 2]
no_decay = [p for _, p in model.named_parameters() if p.ndim < 2]
opt   = torch.optim.SGD([{"params": decay,    "weight_decay": 1e-4},
                          {"params": no_decay, "weight_decay": 0}],
                         lr=LR, momentum=0.9, nesterov=True)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
crit   = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

# Train
best_val = 0; best_sd = None; t0 = time.time()

for ep in range(1, EPOCHS + 1):
    model.train(); ls = n = 0
    for x, y in tr_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE.type, enabled=scaler.is_enabled()):
            loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ls += loss.item() * y.size(0); n += y.size(0)
    tr_loss = ls / n
    sched.step()

    # Val accuracy
    model.eval()
    with torch.no_grad():
        c = t = 0
        for x, y in va_ld:
            c += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item(); t += y.size(0)
        val_acc = c / t

    if val_acc > best_val:
        best_val = val_acc
        best_sd  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if ep % 25 == 0 or ep == 1:
        eta = (time.time() - t0) / ep * (EPOCHS - ep)
        print(f"  ep{ep:3d} | loss={tr_loss:.4f} val={val_acc:.4f} best={best_val:.4f} | ETA {eta/60:.0f}m")

# Test accuracy
model.load_state_dict(best_sd); model.to(DEVICE); model.eval()
with torch.no_grad():
    c = t = 0
    for x, y in te_ld:
        c += (model(x.to(DEVICE)).argmax(1).cpu() == y).sum().item(); t += y.size(0)
    te_acc = c / t

# F1
with torch.no_grad():
    cm = torch.zeros(10, 10, dtype=torch.long)
    for x, y in te_ld:
        p = model(x.to(DEVICE)).argmax(1).cpu()
        for t_, pr in zip(y, p): cm[t_, pr] += 1
f1s = []
for c in range(10):
    tp = cm[c,c].item(); fp = cm[:,c].sum().item()-tp; fn = cm[c,:].sum().item()-tp
    pr_ = tp/(tp+fp+1e-8); rc = tp/(tp+fn+1e-8)
    f1s.append(2*pr_*rc/(pr_+rc+1e-8))
te_f1 = float(np.mean(f1s))

elapsed = time.time() - t0
print(f"\n{'='*55}")
print(f"  ★ plain34 seed=7 FINAL RESULT")
print(f"{'='*55}")
print(f"  Test Accuracy : {te_acc*100:.2f}%")
print(f"  F1 Macro      : {te_f1:.4f}")
print(f"  Best Val      : {best_val*100:.2f}%")
print(f"  Time          : {elapsed/60:.1f} min")
print(f"{'='*55}")
print(f"\nPASTE THIS LINE BACK:")
print(f"  plain34 | seed=7 | test={te_acc*100:.2f}% | f1={te_f1:.4f} | best_val={best_val*100:.2f}%")

result = {"architecture": "plain34", "seed": 7,
          "test_acc": round(te_acc*100, 2), "f1": round(te_f1, 4),
          "best_val": round(best_val*100, 2), "time_min": round(elapsed/60, 1),
          "notes": "real"}

import json
with open("/tmp/plain34_seed7.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved → /tmp/plain34_seed7.json")
