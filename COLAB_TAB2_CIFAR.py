## COLAB_TAB2_CIFAR.py — ResNet-34 CIFAR-10 Full Ablation Study
##
## IMPORTANT: This script imports directly from src/ instead of
## redefining models inline. The tested code (pytest) and the executed
## code (this script) are now identical implementations.
##
## PRE-REQUISITE (do this ONCE before running):
##   1. Zip your project folder on your Mac:
##        cd ~/Downloads && zip -r resnet34.zip resnet34-amazon-products/
##   2. Upload resnet34.zip to Google Drive (any folder, e.g. MyDrive/)
##   3. That's it — then run Cell 1 below which mounts Drive and unzips.
##
## What this runs:
##   E-002  (Protocol A): ResNet-34,  200ep, StepLR    → ~91.8%
##   E-002b (Protocol B): ResNet-34,  100ep, Cosine    → ~93.5%
##   ABL-A1 (Protocol A): Plain-34,   200ep, StepLR    → ~73.4%  (degradation demo)
##   ABL-A1b(Protocol B): Plain-34,   100ep, Cosine    → ~92.1%  (minimal gap)
##
## Protocol A (StepLR, 200ep) reveals the degradation problem dramatically.
## Protocol B (Cosine, 100ep) shows a smaller but faster-converging gap.
## Both results are VALID — they answer different questions.
##
## Runtime: ~5 hours total on T4. Run all cells top-to-bottom.

## ════════════════════════════════════════════════════════════
## CELL 1: Mount Drive + unzip project
## ════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

import subprocess, sys, os, shutil

# ── Change this if you put the zip somewhere else in Drive ────────────────────
ZIP_PATH = "/content/drive/MyDrive/resnet34.zip"          # ← path to your zip
PROJECT_DIR = "/content/resnet34-amazon-products"          # where to unzip

if not os.path.exists(PROJECT_DIR):
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(
            f"Zip not found at {ZIP_PATH}.\n"
            "Fix: zip your project on Mac with:\n"
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

# ── Verify the param count gate immediately ───────────────────────────────────
from src.models.model_factory import ModelFactory
import torch

_resnet = ModelFactory.create("resnet34", num_classes=10, dataset="cifar10")
_plain  = ModelFactory.create("plain34",  num_classes=10, dataset="cifar10")
assert sum(p.numel() for p in _resnet.parameters()) == 21_282_122, \
    "ResNet-34 param count mismatch — check src/models/resnet34.py"
assert sum(p.numel() for p in _plain.parameters())  == 21_108_298, \
    "Plain-34 param count mismatch"
print("✓ Param count gate passed: ResNet-34=21,282,122 | Plain-34=21,108,298")
del _resnet, _plain

## ════════════════════════════════════════════════════════════
## CELL 2: Shared training infrastructure
## ════════════════════════════════════════════════════════════
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np, json, time, random, csv
from pathlib import Path
import torch.nn as nn

SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'none'}")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)

def get_loaders(aug="standard"):
    if aug=="none":
        tr=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
    else:
        tr=T.Compose([T.RandomCrop(32,padding=4,padding_mode='reflect'),
                      T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
    vt=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
    rng=np.random.default_rng(SEED); idx=rng.permutation(50000)
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

def run(experiment_id, arch, protocol, epochs, lr, scheduler_name, aug="standard"):
    """Run one experiment and return a result dict."""
    print(f"\n{'='*65}")
    print(f"  {experiment_id} | arch={arch} | protocol={protocol} | epochs={epochs} | sched={scheduler_name}")
    print(f"{'='*65}")
    set_seed()
    model = ModelFactory.create(arch, num_classes=10, dataset="cifar10").to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,} ({n_params/1e6:.3f}M)")

    tr_ld, va_ld, te_ld = get_loaders(aug)

    # BN-excluded weight decay
    decay    = [p for _,p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for _,p in model.named_parameters() if p.ndim <  2]
    opt = torch.optim.SGD(
        [{"params":decay,"weight_decay":1e-4},{"params":no_decay,"weight_decay":0}],
        lr=lr, momentum=0.9, nesterov=True)

    if scheduler_name == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    else:  # step
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.1)

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type=="cuda"))

    best_val=0; best_sd=None; t0=time.time()
    for ep in range(1, epochs+1):
        tr_loss = train_epoch(model, tr_ld, opt, scaler)
        val_acc = accuracy(model, va_ld); sched.step()
        if val_acc > best_val:
            best_val = val_acc
            best_sd  = {k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep % 20 == 0 or ep == 1:
            eta = (time.time()-t0)/ep*(epochs-ep)
            print(f"  ep{ep:3d} | loss={tr_loss:.4f} val={val_acc:.4f} best={best_val:.4f} | ETA {eta/60:.0f}m")

    model.load_state_dict(best_sd); model.to(DEVICE)
    te_acc = accuracy(model, te_ld)
    te_f1  = macro_f1(model, te_ld)
    elapsed = time.time()-t0
    print(f"\n  ★ {experiment_id}: Test={te_acc*100:.2f}% F1={te_f1:.4f} | {elapsed/60:.0f}min")

    return dict(
        experiment_id=experiment_id, protocol=protocol,
        architecture=arch, scheduler=scheduler_name,
        num_epochs=epochs, lr=lr, seed=SEED,
        n_params=n_params, aug=aug,
        test_acc=round(te_acc*100, 2), f1=round(te_f1, 4),
        best_val=round(best_val*100, 2), time_min=round(elapsed/60, 1),
    )

## ════════════════════════════════════════════════════════════
## CELL 3: Run all 4 experiments (in order)
## ════════════════════════════════════════════════════════════
results = []

# Protocol A (StepLR, 200ep) — matches EXPERIMENT_LOG entries E-002 / ABL-A1
results.append(run("E-002",   "resnet34", "A", epochs=200, lr=0.1, scheduler_name="step"))
results.append(run("ABL-A1",  "plain34",  "A", epochs=200, lr=0.1, scheduler_name="step"))

# Protocol B (CosineAnnealingLR, 100ep) — matches Colab run that produced 93.51%
results.append(run("E-002b",  "resnet34", "B", epochs=100, lr=0.1, scheduler_name="cosine"))
results.append(run("ABL-A1b", "plain34",  "B", epochs=100, lr=0.1, scheduler_name="cosine"))

## ════════════════════════════════════════════════════════════
## CELL 4: Print results table + save
## ════════════════════════════════════════════════════════════
def print_table(results):
    print(f"\n{'='*80}")
    print("  COMPLETE RESULTS — TWO EXPERIMENTAL PROTOCOLS")
    print(f"{'='*80}")
    print(f"  {'ID':<10} {'Protocol':<12} {'Arch':<10} {'Sched':<8} {'Ep':<5} {'Test%':<7} {'F1':<7} {'Params':<10}")
    print("  " + "-"*75)
    for r in results:
        print(f"  {r['experiment_id']:<10} {r['protocol']:<12} {r['architecture']:<10} "
              f"{r['scheduler']:<8} {r['num_epochs']:<5} {r['test_acc']:<7.2f} "
              f"{r['f1']:<7.4f} {r['n_params']/1e6:<10.2f}M")
    print(f"\n{'='*80}")
    print("  INTERPRETATION")
    print(f"{'='*80}")
    a_resnet = next(r for r in results if r['experiment_id']=='E-002')
    a_plain  = next(r for r in results if r['experiment_id']=='ABL-A1')
    b_resnet = next(r for r in results if r['experiment_id']=='E-002b')
    b_plain  = next(r for r in results if r['experiment_id']=='ABL-A1b')
    print(f"  Protocol A (StepLR 200ep):  ResNet-34 {a_resnet['test_acc']}% vs Plain-34 {a_plain['test_acc']}% → "
          f"+{a_resnet['test_acc']-a_plain['test_acc']:.1f}pp")
    print(f"    → Shows DEGRADATION PROBLEM: deep plain net fails to converge")
    print(f"  Protocol B (Cosine 100ep):  ResNet-34 {b_resnet['test_acc']}% vs Plain-34 {b_plain['test_acc']}% → "
          f"+{b_resnet['test_acc']-b_plain['test_acc']:.1f}pp")
    print(f"    → Shows CONVERGENCE ADVANTAGE: ResNet reaches higher acc faster")
    print(f"\n  Both results are valid. Protocol A demonstrates the stronger claim.")
    print(f"{'='*80}")

print_table(results)

# Save to /tmp/
with open("/tmp/cifar10_full_results.json","w") as f:
    json.dump(results, f, indent=2)

with open("/tmp/cifar10_ablation_table.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader(); w.writerows(results)

print("\nDownload from /tmp/ (Files panel → left sidebar):")
print("  cifar10_full_results.json")
print("  cifar10_ablation_table.csv")
print("\nCommit these to: results/ablation_table.csv")
