## PASTE THIS INTO COLAB TAB 2 — ONE CELL
## Runtime → T4 GPU → Run → Walk away (~4 hours)
## Covers tracker Days 24, 32, 33, 34, 35, 36

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np, json, time, random, csv
from pathlib import Path

SEED = 42
def set_seed():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

set_seed()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if DEVICE.type=='cuda' else 'none'}")

# ── Models ─────────────────────────────────────────────────────────
class BasicBlock(nn.Module):
    def __init__(self,ic,oc,stride=1,skip=True,bn=True):
        super().__init__()
        self.skip=skip
        self.conv1=nn.Conv2d(ic,oc,3,stride=stride,padding=1,bias=not bn)
        self.bn1=nn.BatchNorm2d(oc) if bn else nn.Identity()
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(oc,oc,3,padding=1,bias=not bn)
        self.bn2=nn.BatchNorm2d(oc) if bn else nn.Identity()
        self.sc=nn.Identity()
        if skip and (stride!=1 or ic!=oc):
            sc=[nn.Conv2d(ic,oc,1,stride=stride,bias=not bn)]
            if bn: sc.append(nn.BatchNorm2d(oc))
            self.sc=nn.Sequential(*sc)
    def forward(self,x):
        out=self.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        if self.skip: out=out+self.sc(x)
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self,blocks,num_classes=10,skip=True,bn=True):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3,64,3,1,1,bias=not bn),
            nn.BatchNorm2d(64) if bn else nn.Identity(),nn.ReLU(inplace=True))
        self.layer1=self._make(64,  64, blocks[0],1,skip,bn)
        self.layer2=self._make(64, 128, blocks[1],2,skip,bn)
        self.layer3=self._make(128,256, blocks[2],2,skip,bn)
        self.layer4=self._make(256,512, blocks[3],2,skip,bn)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(512,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def _make(self,ic,oc,n,stride,skip,bn):
        return nn.Sequential(BasicBlock(ic,oc,stride,skip,bn),*[BasicBlock(oc,oc,1,skip,bn) for _ in range(1,n)])
    def forward(self,x):
        x=self.stem(x)
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.pool(x),1))

class BasicCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        def blk(ic,oc,pool=True):
            m=[nn.Conv2d(ic,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(inplace=True)]
            if pool: m.append(nn.MaxPool2d(2))
            return nn.Sequential(*m)
        self.net=nn.Sequential(blk(3,64,True),blk(64,128,True),blk(128,256,True),
                               blk(256,512,False),blk(512,512,False),nn.AdaptiveAvgPool2d(1))
        self.fc=nn.Linear(512,num_classes)
        for m in self.modules():
            if isinstance(m,nn.Conv2d): nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d): nn.init.constant_(m.weight,1); nn.init.constant_(m.bias,0)
    def forward(self,x): return self.fc(torch.flatten(self.net(x),1))

# ── Data ───────────────────────────────────────────────────────────
MEAN=(0.4914,0.4822,0.4465); STD=(0.2470,0.2435,0.2616)

def get_loaders(aug="standard"):
    if aug=="none":
        tr=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
    elif aug=="standard":
        tr=T.Compose([T.RandomCrop(32,padding=4,padding_mode='reflect'),
                      T.RandomHorizontalFlip(),T.ToTensor(),T.Normalize(MEAN,STD)])
    else:
        tr=T.Compose([T.RandomCrop(32,padding=4,padding_mode='reflect'),
                      T.RandomHorizontalFlip(),
                      T.ColorJitter(0.4,0.4,0.4,0.1),T.RandomGrayscale(0.1),
                      T.ToTensor(),T.Normalize(MEAN,STD),
                      T.RandomErasing(p=0.5,scale=(0.02,0.2))])
    vt=T.Compose([T.ToTensor(),T.Normalize(MEAN,STD)])
    rng=np.random.default_rng(SEED); idx=rng.permutation(50000)
    tr_ds=torchvision.datasets.CIFAR10('/tmp/c10',True,tr,download=True)
    va_ds=torchvision.datasets.CIFAR10('/tmp/c10',True,vt,download=False)
    te_ds=torchvision.datasets.CIFAR10('/tmp/c10',False,vt,download=False)
    tr_ld=DataLoader(Subset(tr_ds,idx[:45000]),128,shuffle=True, num_workers=2,pin_memory=True)
    va_ld=DataLoader(Subset(va_ds,idx[45000:]),256,shuffle=False,num_workers=2,pin_memory=True)
    te_ld=DataLoader(te_ds,256,shuffle=False,num_workers=2,pin_memory=True)
    return tr_ld,va_ld,te_ld

# ── Training helpers ───────────────────────────────────────────────
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

def run(label, model, aug="standard", epochs=100, lr=0.1):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    set_seed(); model=model.to(DEVICE)
    np=sum(p.numel() for p in model.parameters())
    print(f"Params: {np/1e6:.2f}M | Aug: {aug} | LR: {lr}")
    tr_ld,va_ld,te_ld=get_loaders(aug)
    if lr<=0.01:
        opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,nesterov=True,weight_decay=1e-4)
    else:
        decay=[p for _,p in model.named_parameters() if p.ndim>=2]
        nodecay=[p for _,p in model.named_parameters() if p.ndim<2]
        opt=torch.optim.SGD([{"params":decay,"weight_decay":1e-4},
                              {"params":nodecay,"weight_decay":0}],lr=lr,momentum=0.9,nesterov=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler("cuda",enabled=(DEVICE.type=="cuda"))
    best_val=0;best_sd=None;t0=time.time()
    for ep in range(1,epochs+1):
        tr_loss=train_epoch(model,tr_ld,opt,scaler)
        val_acc=accuracy(model,va_ld); sched.step()
        if val_acc>best_val: best_val=val_acc; best_sd={k:v.cpu().clone() for k,v in model.state_dict().items()}
        if ep%10==0 or ep==1:
            eta=(time.time()-t0)/ep*(epochs-ep)
            print(f"ep{ep:3d} | loss={tr_loss:.4f} val={val_acc:.4f} best={best_val:.4f} | ETA {eta/60:.0f}m")
    model.load_state_dict(best_sd); model.to(DEVICE)
    te_acc=accuracy(model,te_ld); te_f1=macro_f1(model,te_ld)
    elapsed=time.time()-t0
    print(f"\n★ {label}: {te_acc*100:.2f}% | F1={te_f1:.4f} | {elapsed/60:.0f}min")
    return dict(label=label,aug=aug,n_params=np,
                test_acc=round(te_acc*100,2),f1=round(te_f1,4),
                best_val=round(best_val*100,2),time_min=round(elapsed/60,1))

# ── Already have these — paste from your first Colab run ───────────
results=[
    dict(label="ResNet-34 skip=True", aug="standard",n_params=21282122,test_acc=93.51,f1=0.9349,best_val=94.26,time_min=60.3),
    dict(label="Plain-34  skip=False",aug="standard",n_params=21108298,test_acc=92.13,f1=0.9212,best_val=92.38,time_min=56.7),
]

# ── Run remaining experiments ──────────────────────────────────────
results.append(run("BasicCNN (no residuals)",    BasicCNN(),                          aug="standard",epochs=100,lr=0.01))
results.append(run("ResNet-18 (depth ablation)", ResNet([2,2,2,2],skip=True,bn=True), aug="standard",epochs=100,lr=0.1))
results.append(run("ResNet-34 no BatchNorm",     ResNet([3,4,6,3],skip=True,bn=False),aug="standard",epochs=100,lr=0.001))
results.append(run("ResNet-34 no augmentation",  ResNet([3,4,6,3],skip=True,bn=True), aug="none",    epochs=100,lr=0.1))
results.append(run("ResNet-34 heavy aug",        ResNet([3,4,6,3],skip=True,bn=True), aug="heavy",   epochs=100,lr=0.1))

# ── Print complete table ───────────────────────────────────────────
rn34 = next(r for r in results if "skip=True"  in r["label"] and "18" not in r["label"])
pl34 = next(r for r in results if "skip=False" in r["label"])
rn18 = next(r for r in results if "ResNet-18"  in r["label"])
bcnn = next(r for r in results if "BasicCNN"   in r["label"])
nobn = next(r for r in results if "no BatchNorm" in r["label"])
noaug= next(r for r in results if "no augmentation" in r["label"])
hvaug= next(r for r in results if "heavy"      in r["label"])

print(f"\n{'='*70}")
print("  COMPLETE ABLATION TABLE")
print(f"{'='*70}")
print(f"  {'Model':<35} {'Acc%':>7} {'F1':>7} {'Params':>9}")
print("  "+"-"*60)
for r in results:
    print(f"  {r['label']:<35} {r['test_acc']:>7.2f} {r['f1']:>7.4f} {r['n_params']/1e6:>8.2f}M")

print(f"""
  KEY FINDINGS:
  Skip connections:  ResNet-34 {rn34['test_acc']}% vs Plain-34 {pl34['test_acc']}% → +{rn34['test_acc']-pl34['test_acc']:.1f}pp
  Depth:             ResNet-18 {rn18['test_acc']}% vs ResNet-34 {rn34['test_acc']}% → +{rn34['test_acc']-rn18['test_acc']:.1f}pp
  vs Baseline:       BasicCNN  {bcnn['test_acc']}% vs ResNet-34 {rn34['test_acc']}% → +{rn34['test_acc']-bcnn['test_acc']:.1f}pp
  BatchNorm:         no-BN     {nobn['test_acc']}% vs ResNet-34 {rn34['test_acc']}% → -{rn34['test_acc']-nobn['test_acc']:.1f}pp without BN
  Augmentation:      no-aug    {noaug['test_acc']}% vs standard {rn34['test_acc']}% → +{rn34['test_acc']-noaug['test_acc']:.1f}pp from aug
""")

# ── Save ───────────────────────────────────────────────────────────
with open("/tmp/all_results.json","w") as f: json.dump(results,f,indent=2)

with open("/tmp/ablation_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["label","aug","n_params","test_acc","f1","best_val","time_min"])
    w.writeheader(); w.writerows(results)

print("Saved → /tmp/all_results.json")
print("Saved → /tmp/ablation_table.csv")
print("\nDOWNLOAD THESE FROM /tmp/ (Files panel on left)")
