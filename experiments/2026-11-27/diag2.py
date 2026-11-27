exec(open('/tmp/diag.py').read().split("cache={}")[0])
(Xt,yt,pt),(Xv,yv,pv)=load("train",64),load("val",64)
h=lambda p:hashlib.md5(open(p,'rb').read()).hexdigest()
th={h(p) for p in pt}
exact=torch.tensor([h(p) in th for p in pv])
def emb(X):
    e=torch.nn.functional.interpolate(X.float(),16,mode="area").flatten(1); e=e-e.mean(1,keepdim=True); return e/e.norm(dim=1,keepdim=True)
S=emb(Xv)@emb(Xt).T; best,j=S.max(1); dup=best>0.97
m=C.Net(res=64); m.load_state_dict(torch.load("model.torch",weights_only=True)); m.eval()
for n,k in [("all",torch.ones(len(yv),dtype=bool)),("exact-md5 dups",exact),("near dups",dup),("no near dups",~dup)]:
    print(n,int(k.sum()),round(acc(m,Xv[k],yv[k]),4))
print("near-dup label disagreement",int((yt[j][dup]!=yv[dup]).sum()))
with torch.no_grad():
    p=m(norm(Xv)).argmax(1)
for c,name in enumerate(C.CLASSES):
    k=(yv==c)&~dup; print(name,int(k.sum()),round((p[k]==c).float().mean().item(),3))
