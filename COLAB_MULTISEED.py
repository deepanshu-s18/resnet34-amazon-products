## COLAB_MULTISEED.py
## ════════════════════════════════════════════════════════════════
## Multi-seed CIFAR-10 training: 3 seeds × 2 models
## = 6 runs, ~3 hours total on T4 GPU
##
## Purpose: Establish mean ± std for the ResNet-34 vs Plain-34 gap.
## A 1.38pp gap from a single seed is ambiguous (within noise range).
## Three seeds let us ask: is the gap statistically consistent?
##
## Seeds: 42, 123, 7
## Protocol: B (Cosine, 100ep) — faster than Protocol A, reproducible
## Architecture: ResNet-34 and Plain-34, imported from src/
##
## PRE-REQUISITE (do this ONCE before running):
##   1. Zip your project on your Mac:
##        cd ~/Downloads && zip -r resnet34.zip resnet34-amazon-products/
##   2. Upload resnet34.zip to Google Drive root (MyDrive/)
##   Then run Cell 1 below.
##
## Output (download from /tmp/ when done):
##   multi_seed_results.json
##   multi_seed_results.csv
## ════════════════════════════════════════════════════════════════

## CELL 1: Mount Drive + setup
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys, os

# ── Change this path if you uploaded the zip to a subfolder in Drive ──────────
ZIP_PATH    = "/content/drive/MyDrive/resnet34.zip"
PROJECT_DIR = "/content/resnet34-amazon-products"

if not os.path.exists(PROJECT_DIR):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Zip not found at {ZIP_PATH}.\n"
            "On your Mac, run:\n"
            "  cd ~/Downloads && zip -r resnet34.zip resnet34-amazon-products/\n"
            "Then upload resnet34.zip to your Google Drive root."
        )
    print(f"Unzipping {ZIP_PATH} → /content/ ...")
    subprocess.run(["unzip", "-q", ZIP_PATH, "-d", "/content/"], check=True)
    print("✓ Unzipped")
else:
    print("✓ Project directory already exists (re-using)")

sys.path.insert(0, PROJECT_DIR)
subprocess.run(["pip", "install", "-e", PROJECT_DIR, "-q"], check=True)
print("✓ Package installed from src/")


from src.models.model_factory import ModelFactory
import torch, torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import numpy as np, json, time, random, csv
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'none'}")

SEEDS    = [42, 123, 7]
EPOCHS   = 100
LR       = 0.1
MEAN     = (0.4914,0.4822,0.4465)
STD      = (0.2470,0.2435,0.2616)

# Param count gate
for arch, expected in [("resnet34", 21_282_122), ("plain34", 21_108_298)]:
    m = ModelFactory.create(arch, num_classes=10, dataset="cifar10")
    actual = sum(p.numel() for p in m.parameters())
    assert actual == expected, f"{arch}: expected {expected:,} params, got {actual:,}"
print("✓ Param count gates passed")

## CELL 2: Shared helpers
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def get_loaders(seed):
    tr=T.Compose([T.RandomCrop(32,padding=4,padding_mode='reflect'),
                  T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
    vt=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
    rng=np.random.default_rng(seed); idx=rng.permutation(50000)
    tr_ds=torchvision.datasets.CIFAR10('/tmp/c10',True,tr,download=True)
    va_ds=torchvision.datasets.CIFAR10('/tmp/c10',True,vt,download=False)
    te_ds=torchvision.datasets.CIFAR10('/tmp/c10',False,vt,download=False)
    return (DataLoader(Subset(tr_ds,idx[:45000]),128,shuffle=True, num_workers=2,pin_memory=True),
            DataLoader(Subset(va_ds,idx[45000:]),256,shuffle=False,num_workers=2,pin_memory=True),
            DataLoader(te_ds,256,shuffle=False,num_workers=2,pin_memory=True))

def train_epoch(model,loader,opt,scaler):
    model.train(); crit=nn.CrossEntropyLoss(); ls=n=0
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(DEVICE.type,enabled=scaler.is_enabled()):
            loss=crit(model(x),y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ls+=loss.item()*y.size(0); n+=y.size(0)
    return ls/n

@torch.no_grad()
def accuracy(model,loader):
    model.eval(); c=t=0
    for x,y in loader:
        c+=(model(x.to(DEVICE)).argmax(1).cpu()==y).sum().item(); t+=y.size(0)
    return c/t

@torch.no_grad()
def macro_f1(model,loader,K=10):
    model.eval(); cm=torch.zeros(K,K,dtype=torch.long)
    for x,y in loader:
        p=model(x.to(DEVICE)).argmax(1).cpu()
        for t,pr in zip(y,p): cm[t,pr]+=1
    f1s=[]
    for c in range(K):
        tp=cm[c,c].item();fp=cm[:,c].sum().item()-tp;fn=cm[c,:].sum().item()-tp
        p_=tp/(tp+fp+1e-8);r=tp/(tp+fn+1e-8);f1s.append(2*p_*r/(p_+r+1e-8))
    return float(np.mean(f1s))

def run_one(arch, seed):
    """Run one (arch, seed) combination. Returns result dict."""
    label = f"{arch} seed={seed}"
    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")

    set_seed(seed)
    model = ModelFactory.create(arch, num_classes=10, dataset="cifar10").to(DEVICE)
    tr_ld, va_ld, te_ld = get_loaders(seed)

    decay    = [p for _,p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for _,p in model.named_parameters() if p.ndim <  2]
    opt   = torch.optim.SGD([{"params":decay,"weight_decay":1e-4},
                              {"params":no_decay,"weight_decay":0}],
                             lr=LR, momentum=0.9, nesterov=True)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type=="cuda"))

    best_val=0; best_sd=None; t0=time.time()
    for ep in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, tr_ld, opt, scaler)
        val_acc = accuracy(model, va_ld); sched.step()
        if val_acc > best_val:
            best_val = val_acc
            best_sd  = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 25 == 0 or ep == 1:
            eta=(time.time()-t0)/ep*(EPOCHS-ep)
            print(f"  ep{ep:3d} | loss={tr_loss:.4f} val={val_acc:.4f} | ETA {eta/60:.0f}m")

    model.load_state_dict(best_sd); model.to(DEVICE)
    te_acc = accuracy(model, te_ld)
    te_f1  = macro_f1(model, te_ld)
    elapsed = time.time()-t0
    print(f"  ★ {label}: {te_acc*100:.2f}% F1={te_f1:.4f} | {elapsed/60:.0f}min")
    return dict(architecture=arch, seed=seed,
                test_acc=round(te_acc*100,2), f1=round(te_f1,4),
                best_val=round(best_val*100,2), time_min=round(elapsed/60,1))

## CELL 3: Run all 6 combinations (resnet34 × 3 seeds, then plain34 × 3 seeds)
all_results = []
for seed in SEEDS:
    all_results.append(run_one("resnet34", seed))
    all_results.append(run_one("plain34",  seed))

## CELL 4: Compute statistics + print final table
def stats(values):
    a = np.array(values)
    return round(float(a.mean()),2), round(float(a.std()),2)

resnet_accs = [r["test_acc"] for r in all_results if r["architecture"]=="resnet34"]
plain_accs  = [r["test_acc"] for r in all_results if r["architecture"]=="plain34"]
resnet_f1s  = [r["f1"]       for r in all_results if r["architecture"]=="resnet34"]
plain_f1s   = [r["f1"]       for r in all_results if r["architecture"]=="plain34"]

res_mean, res_std = stats(resnet_accs)
pla_mean, pla_std = stats(plain_accs)
gap_mean = round(res_mean - pla_mean, 2)
gap_stds = [r-p for r,p in zip(resnet_accs, plain_accs)]
gap_std  = round(float(np.array(gap_stds).std()),2)

print(f"\n{'='*65}")
print("  MULTI-SEED RESULTS SUMMARY (Protocol B: Cosine, 100ep)")
print(f"{'='*65}")
print(f"\n  Per-run results:")
print(f"  {'Arch':<12} {'Seed':<6} {'Test%':<8} {'F1':<7}")
print(f"  {'-'*35}")
for r in all_results:
    print(f"  {r['architecture']:<12} {r['seed']:<6} {r['test_acc']:<8.2f} {r['f1']:<7.4f}")

print(f"\n  Aggregate:")
print(f"  ResNet-34:  {res_mean:.2f}% ± {res_std:.2f}%   F1={np.mean(resnet_f1s):.4f}")
print(f"  Plain-34:   {pla_mean:.2f}% ± {pla_std:.2f}%   F1={np.mean(plain_f1s):.4f}")
print(f"  Gap:        {gap_mean:.2f}pp ± {gap_std:.2f}pp")
print(f"\n  Interpretation:")
if gap_std < gap_mean / 2:
    print(f"  Gap is CONSISTENT across seeds (std < mean/2) ✓")
    print(f"  The {gap_mean:.2f}pp advantage is reproducible, not a fluke.")
else:
    print(f"  Gap is VARIABLE across seeds (std ≥ mean/2) ⚠")
    print(f"  Run more seeds or check for implementation issues.")
print(f"{'='*65}")

## CELL 5: Save results
summary = {
    "protocol": "B (CosineAnnealingLR, 100 epochs, lr=0.1)",
    "seeds": SEEDS,
    "resnet34": {"accs": resnet_accs, "f1s": resnet_f1s,
                 "mean_acc": res_mean, "std_acc": res_std},
    "plain34":  {"accs": plain_accs,  "f1s": plain_f1s,
                 "mean_acc": pla_mean, "std_acc": pla_std},
    "gap_pp":   {"values": gap_stds, "mean": gap_mean, "std": gap_std},
    "per_run": all_results,
}

with open("/tmp/multi_seed_results.json","w") as f:
    json.dump(summary, f, indent=2)
print("Saved → /tmp/multi_seed_results.json")

fieldnames = list(all_results[0].keys())
with open("/tmp/multi_seed_results.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fieldnames)
    w.writeheader(); w.writerows(all_results)
print("Saved → /tmp/multi_seed_results.csv")

print(f"""
{'='*65}
  PASTE THESE LINES INTO README.md results table:

  | ResNet-34 | {res_mean:.2f}% ± {res_std:.2f}% | Protocol B, 3 seeds |
  | Plain-34  | {pla_mean:.2f}% ± {pla_std:.2f}% | Protocol B, 3 seeds |
  | **Gap**   | **{gap_mean:.2f}pp ± {gap_std:.2f}pp** | skip connections benefit |
{'='*65}
""")
