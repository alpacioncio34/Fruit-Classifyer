# Ablation harness (read-only on the repo). Reports the LAST-epoch EMA, no epoch selection.
import sys, math, copy, time, hashlib, io, os
sys.dont_write_bytecode=True; sys.path.insert(0,"/home/jovyan/shared/HLCV-Challenge"); os.chdir("/home/jovyan/shared/HLCV-Challenge")
import torch, torch.nn as nn, torch.nn.functional as Fn, torchvision, torchvision.transforms.v2 as v2, torchvision.transforms.v2.functional as F
import challenge_train as C
arch,aug,epochs,seed=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
torch.set_num_threads(int(os.environ.get("NT","6"))); torch.manual_seed(seed)
def load(split,res):
    d=torchvision.datasets.ImageFolder(f"dataset/{split}",transform=v2.Compose([v2.ToImage(),v2.Resize(res),v2.CenterCrop(res)]))
    return torch.stack([x.as_subclass(torch.Tensor) for x,_ in d]),torch.tensor(d.targets)
Xt,yt=load("train",96); Xv,yv=load("val",64); T=torch.load("teacher_logits.pt",weights_only=True)["logits"]
def emb(X):
    e=Fn.interpolate(X.float(),16,mode="area").flatten(1); e=e-e.mean(1,keepdim=True); return e/e.norm(dim=1,keepdim=True)
keep=(emb(Xv)@emb(load("train",64)[0]).T).max(1).values<=0.97; Xv,yv=Xv[keep],yv[keep]
mean=torch.tensor(C.MEAN).view(1,3,1,1); std=torch.tensor(C.STD).view(1,3,1,1)

class DS(nn.Module):   # depthwise separable
    def __init__(s,i,o,st=1):
        super().__init__(); s.res=(i==o and st==1)
        s.f=nn.Sequential(nn.Conv2d(i,i,3,st,1,groups=i,bias=False),nn.BatchNorm2d(i),nn.ReLU(),nn.Conv2d(i,o,1,bias=False),nn.BatchNorm2d(o))
    def forward(s,x): return Fn.relu(x+s.f(x) if s.res else s.f(x))
class MB(nn.Module):   # inverted bottleneck
    def __init__(s,i,e,o,st=1):
        super().__init__(); s.res=(i==o and st==1)
        s.f=nn.Sequential(nn.Conv2d(i,e,1,bias=False),nn.BatchNorm2d(e),nn.SiLU(),nn.Conv2d(e,e,3,st,1,groups=e,bias=False),nn.BatchNorm2d(e),nn.SiLU(),nn.Conv2d(e,o,1,bias=False),nn.BatchNorm2d(o))
    def forward(s,x): return x+s.f(x) if s.res else s.f(x)
def stem(c): return [nn.Conv2d(3,c,3,2,1,bias=False),nn.BatchNorm2d(c),nn.ReLU()]
def head(c): return [nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(0.1),nn.Linear(c,10)]
if arch=="resnet": net=C.Net(res=64)
elif arch=="ds": net=nn.Sequential(*stem(16),DS(16,32,2),DS(32,32),DS(32,32),DS(32,48,2),DS(48,48),*head(48))
elif arch=="mb": net=nn.Sequential(*stem(12),MB(12,24,16,2),MB(16,32,16),MB(16,48,24,2),MB(24,48,24),nn.Conv2d(24,40,1,bias=False),nn.BatchNorm2d(40),nn.SiLU(),*head(40))
n=sum(p.numel() for p in net.parameters()); assert n<=10000

def augment(x):   # x: uint8 [B,3,96,96] -> normalised [B,3,64,64]
    B=len(x); x=x.float()/255; r=lambda a,b:torch.rand(B)*(b-a)+a
    strong=aug!="base"
    if strong: s=torch.exp(r(math.log(0.55),math.log(1.5))); rot=r(-30,30)*math.pi/180*(torch.rand(B)<0.7); ar=torch.exp(r(-0.22,0.22))
    else: s=r(0.92,1.0); rot=torch.zeros(B); ar=torch.exp(r(-0.05,0.05))
    sx,sy=s*ar.sqrt(),s/ar.sqrt()                      # s = part of the source seen (>1: zoom out)
    flip=(torch.rand(B)<0.5).float()*-2+1
    tx=r(-1,1)*(1-sx).abs(); ty=r(-1,1)*(1-sy).abs()
    M=torch.stack([torch.stack([flip*sx*rot.cos(),-sy*rot.sin(),tx],1),torch.stack([flip*sx*rot.sin(),sy*rot.cos(),ty],1)],1)
    g=Fn.affine_grid(M,[B,3,64,64],align_corners=False)
    x=1-Fn.grid_sample(1-x,g,mode="bilinear",padding_mode="zeros",align_corners=False)   # white fill
    j=0.4 if strong else 0.2; u=lambda:r(1-j,1+j).view(B,1,1,1)
    x=(x*u()).clamp(0,1)
    m=x.mean((1,2,3),keepdim=True); x=((x-m)*u()+m).clamp(0,1)
    gray=(x*torch.tensor([0.299,0.587,0.114]).view(1,3,1,1)).sum(1,keepdim=True); x=(gray+(x-gray)*u()).clamp(0,1)
    if strong:
        h=r(-0.15,0.15)*(torch.rand(B)<0.5); c,sn=h.cos(),h.sin(); k=1/3; q=math.sqrt(k)   # small hue rotation about the grey axis
        a=c+(1-c)*k; b=k*(1-c)-q*sn; d=k*(1-c)+q*sn
        R=torch.stack([torch.stack([a,b,d],1),torch.stack([d,a,b],1),torch.stack([b,d,a],1)],1)
        x=torch.einsum("bij,bjhw->bihw",R,x).clamp(0,1)
        gray=x.mean(1,keepdim=True); x=torch.where((torch.rand(B)<0.07).view(B,1,1,1),gray.expand_as(x),x)
        bl=torch.rand(B)<0.25
        if bl.any(): x[bl]=F.gaussian_blur(x[bl],[5,5],[float(torch.rand(1))+0.5]*2)
        x=(x+torch.randn_like(x)*(r(0,0.06)*(torch.rand(B)<0.3)).view(B,1,1,1)).clamp(0,1)
    x=(x-mean)/std
    for i in range(B):
        if torch.rand(1)<(0.5 if strong else 0.3):
            h,w=torch.randint(8,24,(2,)).tolist(); yy=int(torch.randint(0,64-h,(1,))); xx=int(torch.randint(0,64-w,(1,))); x[i,:,yy:yy+h,xx:xx+w]=0
    return x

decay=[p for p in net.parameters() if p.ndim>1]; nod=[p for p in net.parameters() if p.ndim<=1]
lr=1e-3 if arch=="resnet" else 6e-3
wd=0.05 if arch=="resnet" else 0.02
opt=torch.optim.AdamW([{"params":decay,"weight_decay":wd},{"params":nod,"weight_decay":0}],lr=lr)
BS=128; spe=math.ceil(len(yt)/BS); tot=epochs*spe; warm=max(1,int(.03*tot))
sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:(s+1)/warm if s<warm else .01+.99*.5*(1+math.cos(math.pi*(s-warm)/max(1,tot-warm))))
ema=C.EMA(net,0.99); t0=time.time()
mix=aug=="strongmix"
for ep in range(epochs):
    net.train()
    for idx in torch.randperm(len(yt)).split(BS):
        x=augment(Xt[idx]); y=yt[idx]; t=Fn.softmax(T[idx]/4,1); hard=Fn.one_hot(y,10).float()
        if mix and torch.rand(1)<0.5:      # CutMix
            perm=torch.randperm(len(idx)); lam=float(torch.distributions.Beta(1.,1.).sample()); side=int(64*math.sqrt(1-lam))
            if side>0:
                a=int(torch.randint(0,65-side,(1,))); b=int(torch.randint(0,65-side,(1,))); x=x.clone(); x[:,:,a:a+side,b:b+side]=x[perm,:,a:a+side,b:b+side]
                l=1-side*side/4096; t=l*t+(1-l)*t[perm]; hard=l*hard+(1-l)*hard[perm]
        o=net(x); lp=Fn.log_softmax(o/4,1)
        L=0.3*-(hard*Fn.log_softmax(o,1)).sum(1).mean()+0.7*16*Fn.kl_div(lp,t,reduction="batchmean")
        opt.zero_grad(); L.backward(); opt.step(); sch.step(); ema.update(net)
    if (ep+1)%50==0 or ep==0:
        with torch.no_grad(): a=(ema.model((Xv.float()/255-mean)/std).argmax(1)==yv).float().mean()
        print(f"{arch} {aug} s{seed} ep{ep+1} loss {L.item():.3f} val {a:.3f} t {time.time()-t0:.0f}s",flush=True)
m=ema.model.eval()
@torch.no_grad()
def acc(X,tta=False):
    z=(X.float()/255-mean)/std; o=m(z).softmax(1)
    if tta: o=o+m(z.flip(-1)).softmax(1)
    return (o.argmax(1)==yv).float().mean().item()
def pad(s): return lambda x:F.pad(F.resize(x,[s,s],antialias=True),[(64-s)//2]*4,fill=255)
def zoom(s): return lambda x:F.resize(F.center_crop(x,[s,s]),[64,64],antialias=True)
torch.manual_seed(123)
probes={ # "covered": inside the strong augmentation ranges
 "zoom1.33":zoom(48),"pad48":pad(48),"rot25":lambda x:F.rotate(x,25,interpolation=v2.InterpolationMode.BILINEAR,fill=255),
 "bright.65":lambda x:F.adjust_brightness(x,.65),"sat.6":lambda x:F.adjust_saturation(x,.6),"blur":lambda x:F.gaussian_blur(x,[5,5],[1.2,1.2]),"noise":lambda x:(x.float()+torch.randn(x.shape)*12).clamp(0,255).to(torch.uint8)}
unseen={ # never used in any augmentation
 "jpeg20":lambda x:torch.stack([F.jpeg(i,20) for i in x]),"persp":lambda x:F.perspective(x,[[0,0],[63,0],[63,63],[0,63]],[[8,4],[55,0],[63,63],[0,55]],fill=255),
 "greybg":lambda x:torch.where(x.amin(1,keepdim=True)>235,torch.tensor(110,dtype=torch.uint8),x),"warm":lambda x:(x.float()*torch.tensor([1.15,1.0,.75]).view(1,3,1,1)).clamp(0,255).to(torch.uint8),
 "posterize3":lambda x:F.posterize(x,3),"lowres24":lambda x:F.resize(F.resize(x,[24,24],antialias=True),[64,64]),"zoom2":zoom(32)}
c={k:acc(f(Xv)) for k,f in probes.items()}; u={k:acc(f(Xv)) for k,f in unseen.items()}
print(f"RESULT {arch:6s} {aug:9s} ep{epochs} s{seed} params {n} clean {acc(Xv):.3f} tta {acc(Xv,True):.3f} covered {sum(c.values())/len(c):.3f} unseen {sum(u.values())/len(u):.3f} (n_val {len(yv)})",flush=True)
print("   ",{k:round(v,3) for k,v in {**c,**u}.items()},flush=True)
torch.save(m.state_dict(),f"/tmp/{arch}_{aug}_{epochs}_{seed}.pt")
