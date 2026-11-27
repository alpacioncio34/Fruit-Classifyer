import sys, glob, hashlib
sys.dont_write_bytecode=True
sys.path.insert(0,"/home/jovyan/shared/HLCV-Challenge")
import os; os.chdir("/home/jovyan/shared/HLCV-Challenge")
import torch, torchvision, torchvision.transforms.v2 as v2, torchvision.transforms.v2.functional as F
import challenge_train as C
torch.set_num_threads(6)
def load(split,res):
    d=torchvision.datasets.ImageFolder(f"dataset/{split}",transform=v2.Compose([v2.ToImage(),v2.Resize(res),v2.CenterCrop(res)]))
    X=torch.stack([x.as_subclass(torch.Tensor) for x,_ in d]); return X,torch.tensor(d.targets),[p for p,_ in d.samples]
norm=v2.Compose([v2.ToDtype(torch.float32,scale=True),v2.Normalize(C.MEAN,C.STD)])
@torch.no_grad()
def acc(m,X,y,f=lambda x:x,tta=False):
    z=norm(f(X)); o=m(z).softmax(1)
    if tta: o=o+m(z.flip(-1)).softmax(1)
    return (o.argmax(1)==y).float().mean().item()
cache={}
for w,res in [("model.torch",64),("model.torch",16),("model_r64_s1_kd1.torch",64),("model_r16_s0_kd1.torch",16)]:
    sd=torch.load(w,weights_only=True)
    m=C.Net(res=res); r=m.load_state_dict(sd,strict=False); m.eval()
    if res not in cache: cache[res]=(load("train",res),load("val",res))
    (Xt,yt,_),(Xv,yv,_)=cache[res]
    print(w,res,r,"train %.3f val %.3f val+flip %.3f"%(acc(m,Xt,yt),acc(m,Xv,yv),acc(m,Xv,yv,tta=True)),flush=True)
# robustness probes
m=C.Net(res=64); m.load_state_dict(torch.load("model.torch",weights_only=True)); m.eval()
X,y,_=cache[64][1]
pad=lambda s:(lambda x:F.pad(F.resize(x,[s,s],antialias=True),[(64-s)//2]*4,fill=255))
probes={"clean":lambda x:x,
 "zoom_in 1.33x":lambda x:F.resize(F.center_crop(x,[48,48]),[64,64],antialias=True),
 "zoom_in 2x":lambda x:F.resize(F.center_crop(x,[32,32]),[64,64],antialias=True),
 "zoom_out white pad 0.75":pad(48),"zoom_out white pad 0.5":pad(32),
 "rot 30":lambda x:F.rotate(x,30,interpolation=v2.InterpolationMode.BILINEAR,fill=255),
 "bright 0.6":lambda x:F.adjust_brightness(x,0.6),"bright 1.4":lambda x:F.adjust_brightness(x,1.4),
 "saturation 0.5":lambda x:F.adjust_saturation(x,0.5),"hue +0.05":lambda x:F.adjust_hue(x,0.05),
 "grayscale":lambda x:F.rgb_to_grayscale(x,3),"blur":lambda x:F.gaussian_blur(x,[5,5],[1.2,1.2]),
 "white->random grey bg":lambda x:torch.where((x.amin(1,keepdim=True)>235),torch.tensor(110,dtype=torch.uint8),x),
 "noise":lambda x:(x.float()+torch.randn(x.shape)*15).clamp(0,255).to(torch.uint8),
}
for k,f in probes.items(): print(f"probe {k:28s} {acc(m,X,y,f):.3f}")
# accuracy white-bg vs not
white=torch.tensor([ ((torch.cat([a[:,0],a[:,-1],a[:,:,0],a[:,:,-1]],1).amin(0)>235).float().mean()>0.8).item() for a in X])
for n,mask in [("white-bg",white),("other-bg",~white)]: print(n,int(mask.sum()),acc(m,X[mask],y[mask]))
# near-duplicates train vs val (normalised 16x16 grey+colour correlation)
(Xt,yt,pt),(Xv,yv,pv)=cache[16]
def emb(X):
    e=X.float().flatten(1); e=e-e.mean(1,keepdim=True); return e/e.norm(dim=1,keepdim=True)
S=emb(Xv)@emb(Xt).T; best=S.max(1).values
for t in (0.999,0.99,0.97,0.95): print("val images with train near-dup >",t,int((best>t).sum()),"of",len(best))
