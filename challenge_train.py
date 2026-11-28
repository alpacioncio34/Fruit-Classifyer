# -*- coding: utf-8 -*-
"""

    This is an example implementation of what you have to submit in order
    to test your model on the "test" split.

    One command does everything:   python challenge_train.py
      1. builds the distillation soft labels (teacher_logits.pt) if they are missing or stale,
      2. trains the network and saves model.torch,
      3. prints the clean train/val accuracy.
    Then  python challenge_test.py  evaluates model.torch exactly as the challenge does.

"""

from challenge_test import Test
import torch, torchvision, torchvision.transforms.v2 as v2
import numpy as np
import time
import tqdm
from torchvision.ops import StochasticDepth
import copy
import math
import hashlib




# VARIABLES DEFINITION
TRAIN_DATA_DIR = "dataset/train/"
VAL_DATA_DIR = "dataset/val/"
BATCH_SIZE = 128
# Architecture, chosen by an ablation of 22 designs under 10k parameters:
#   "ghost"  = GhostNet-style bottlenecks (winner), "resnet" = previous Mini-ResNet.
ARCH = "ghost"
# Input resolution. ghost: 64 (or 96: same weights count, a max-pool after the stem). resnet: 16/32/64/128.
RES = 64
SEED = 1
# Train augmentation:
#   "gpu_strong" = batch augmentation on the GPU used in the ablation (affine zoom/rotation/aspect, colour,
#                  hue, grey, blur, noise, erasing). It was the largest single gain on distorted images.
#   "torchvision_strong" / "torchvision_base" = previous per-image pipelines.
AUG_MODE = "gpu_strong"
STRONG_AUG = AUG_MODE != "torchvision_base"
KERNEL_SIZE = 3
EPOCH_NUM = 600   # ablation: 600 beat 300 epochs (+1.6 clean val, +2.7 on distorted val)
CLASSES = ["apple","avocado","banana","cherry","kiwi","mango","orange","pineapple","strawberry","watermelon"]
USE_GPU = True
LR = 6e-3          # ablation setting (the Mini-ResNet used 1e-3)
WEIGHT_DECAY = 0.02
DROPOUT = 0.1
# "last" = save the EMA weights of the last epoch (what the ablation measured; val is not used to pick weights).
# "best" = previous behaviour: the epoch with the best val accuracy (optimistic val number).
SELECT = "last"
LAMBDA_REG = 0.01
# EMA of the weights. None disables it. The EMA averages roughly the last 1/(1-decay) optimizer steps
# (with 18 steps/epoch, 0.99 ~ last 100 steps ~ 5-6 epochs).
EMA_DECAY = 0.99
# Knowledge distillation. Loss = ALPHA * CE(hard label) + (1-ALPHA) * T^2 * KL(teacher_T || student_T).
# Teacher = frozen ImageNet-pretrained backbone + linear head; every train image is labelled by a head that was
# NOT fitted on it (K-fold out-of-fold), so the teacher has not memorised it. Built automatically by
# build_teacher_logits() and cached in TEACHER_FILE. The teacher is only used for training and does not count
# towards the 10k parameter budget (check the challenge rules allow a pretrained teacher).
DISTILL = True
TEACHER_FILE = "teacher_logits.pt"
TEACHER_BACKBONE = "convnext_tiny"   # torchvision ImageNet weights, downloaded on first use
TEACHER_RES = 224
TEACHER_FOLDS = 5
TEACHER_HEAD_EPOCHS = 300
TEACHER_HEAD_WD = 1e-3
TEACHER_SEED = 0
TEMPERATURE = 4.0
ALPHA = 0.3     # weight of the hard-label loss
# Calculated through commented code in datatrain
MEAN = [0.6492, 0.5511, 0.402 ]
STD = [0.2646, 0.2748, 0.3047]

def make_transforms(res):
    '''(deterministic eval part, deterministic train part, random train part, eval part). The deterministic part is the same every epoch, so it is cached.'''
    base = [v2.ToImage(), v2.Resize(res), v2.CenterCrop(res)]
    norm = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=MEAN, std=STD)]
    if AUG_MODE == "gpu_strong":
        # uint8 images at 1.5x resolution; gpu_augment() does everything else on whole batches
        big = res * 3 // 2
        return base, [v2.ToImage(), v2.Resize(big), v2.CenterCrop(big)], [v2.Identity()], norm
    if STRONG_AUG:
        # Random crops are cut from a larger cached image, so zoomed-in crops show real detail instead of upscaled pixels
        big = res * 3 // 2
        base_train = [v2.ToImage(), v2.Resize(big), v2.CenterCrop(big)]
        random_part = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(15, interpolation=v2.InterpolationMode.BILINEAR, fill=255),   # fill: white corners
            v2.RandomResizedCrop(res, scale=(0.5, 1.0), ratio=(0.8, 1.25), antialias=True),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),   # hue stays small: colour identifies fruit
            *norm,
            v2.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ]
        return base, base_train, random_part, norm
    random_part = [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(res, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        *norm,
        v2.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    ]
    return base, base, random_part, norm


class CachedDataset(torch.utils.data.Dataset):
    '''Keeps the resized uint8 images in memory and applies only `transform` on access.'''

    def __init__(self, root, base, transform):
        folder = torchvision.datasets.ImageFolder(root=root, transform=v2.Compose(base))
        self.samples = folder.samples
        self.targets = folder.targets
        self.images = torch.stack([x.as_subclass(torch.Tensor) for x, _ in folder])
        self.transform = v2.Compose(transform)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


# Debugging class for sizes
class PrintSize(torch.nn.Module):
    def forward(self,x):
        print(f"The size after flattening is: {x.shape}")
        return x

class ResBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, sd_prob):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_ch)
        if in_ch != out_ch:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = torch.nn.Identity()
        self.sd = StochasticDepth(p=sd_prob, mode="row")

    def forward(self, x):
        out = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(x) + self.sd(out)   # only the residual branch can be dropped
        return torch.nn.functional.relu(out)


#   scale lo/hi, rotation (deg), p_rotation, log aspect range, jitter, hue (rad), p_grey, p_blur, noise std, p_noise, p_erase
GPU_STRONG = (0.55, 1.5, 30, 0.7, 0.22, 0.4, 0.15, 0.07, 0.25, 0.06, 0.3, 0.5)


def gpu_augment(x, res, params=GPU_STRONG):
    '''Batch augmentation on the device of x (uint8, B x 3 x 1.5res x 1.5res) -> normalized float B x 3 x res x res.
    Same code as experiments/2026-11-27/sweep/train.py (aug="strong").'''
    F = torch.nn.functional
    lo, hi, rot_d, p_rot, ar_r, j, hue, p_gray, p_blur, nz, p_nz, p_er = params
    dev = x.device; B = len(x); R = res; x = x.float() / 255
    r = lambda a, b: torch.rand(B, device=dev) * (b - a) + a
    p = lambda q: torch.rand(B, device=dev) < q
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    # geometry: zoom, aspect, rotation, flip, shift; the white background is kept outside the image
    s = torch.exp(r(math.log(lo), math.log(hi))); rot = r(-rot_d, rot_d) * math.pi / 180 * p(p_rot); ar = torch.exp(r(-ar_r, ar_r))
    sx, sy = s * ar.sqrt(), s / ar.sqrt()
    flip = p(0.5).float() * -2 + 1
    tx = r(-1, 1) * (1 - sx).abs(); ty = r(-1, 1) * (1 - sy).abs()
    M = torch.stack([torch.stack([flip * sx * rot.cos(), -sy * rot.sin(), tx], 1), torch.stack([flip * sx * rot.sin(), sy * rot.cos(), ty], 1)], 1)
    g = F.affine_grid(M, [B, 3, R, R], align_corners=False)
    x = 1 - F.grid_sample(1 - x, g, mode="bilinear", padding_mode="zeros", align_corners=False)
    # colour: brightness, contrast, saturation, hue rotation, grey
    u = lambda: r(1 - j, 1 + j).view(B, 1, 1, 1)
    x = (x * u()).clamp(0, 1)
    m = x.mean((1, 2, 3), keepdim=True); x = ((x - m) * u() + m).clamp(0, 1)
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True); x = (gray + (x - gray) * u()).clamp(0, 1)
    if hue:
        h = r(-hue, hue) * p(0.5); c, sn = h.cos(), h.sin(); k = 1 / 3; q = math.sqrt(k)
        a = c + (1 - c) * k; b = k * (1 - c) - q * sn; d = k * (1 - c) + q * sn
        Rm = torch.stack([torch.stack([a, b, d], 1), torch.stack([d, a, b], 1), torch.stack([b, d, a], 1)], 1)
        x = torch.einsum("bij,bjhw->bihw", Rm, x).clamp(0, 1)
    if p_gray:
        x = torch.where(p(p_gray).view(B, 1, 1, 1), x.mean(1, keepdim=True).expand_as(x), x)
    # blur and noise
    bl = p(p_blur)
    if bl.any(): x[bl] = v2.functional.gaussian_blur(x[bl], [5, 5], [float(torch.rand(1)) + 0.5] * 2)
    if p_nz: x = (x + torch.randn_like(x) * (r(0, nz) * p(p_nz)).view(B, 1, 1, 1)).clamp(0, 1)
    x = (x - mean) / std
    # random erasing (to 0 = the mean colour)
    h = r(R / 8, R * 3 / 8).view(B, 1); w = r(R / 8, R * 3 / 8).view(B, 1)
    y0 = torch.rand(B, 1, device=dev) * (R - h); x0 = torch.rand(B, 1, device=dev) * (R - w); t = torch.arange(R, device=dev).float().view(1, R)
    rows = (t >= y0) & (t < y0 + h) & (torch.rand(B, 1, device=dev) < p_er); cols = (t >= x0) & (t < x0 + w)
    return x.masked_fill((rows[:, :, None] & cols[:, None, :]).unsqueeze(1), 0.0)


def conv_bn(i, o, k=3, s=1, groups=1, act=True):
    '''Conv (no bias) + BatchNorm (+ SiLU).'''
    layers = [torch.nn.Conv2d(i, o, k, s, k // 2, groups=groups, bias=False), torch.nn.BatchNorm2d(o)]
    if act:
        layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


class Ghost(torch.nn.Module):
    '''Ghost module: a 1x1 conv makes half of the output channels, a cheap depthwise 3x3 makes the other half from them.'''

    def __init__(self, i, o, act=True):
        super().__init__()
        h = o // 2
        self.p = conv_bn(i, h, 1, act=act)
        self.c = conv_bn(h, h, 3, 1, groups=h, act=act)

    def forward(self, x):
        y = self.p(x)
        return torch.cat([y, self.c(y)], 1)


class GhostBottleneck(torch.nn.Module):
    '''Ghost (expand) -> [depthwise stride-2 conv] -> Ghost (project, linear). Residual when the shape is unchanged.'''

    def __init__(self, i, e, o, stride=1):
        super().__init__()
        self.res = (i == o and stride == 1)
        layers = [Ghost(i, e)]
        if stride > 1:
            layers.append(conv_bn(e, e, 3, stride, groups=e, act=False))
        layers.append(Ghost(e, o, act=False))
        self.f = torch.nn.Sequential(*layers)

    def forward(self, x):
        return x + self.f(x) if self.res else self.f(x)


class Head(torch.nn.Module):
    def __init__(self, c, dropout):
        super().__init__()
        self.d = torch.nn.Dropout(dropout)
        self.l = torch.nn.Linear(c, len(CLASSES))

    def forward(self, x):
        return self.l(self.d(x.mean((2, 3))))


class GhostNet(torch.nn.Sequential):
    '''
        ARCHITECTURE CHOSEN BY THE ABLATION: tiny GhostNet (9626 parameters).

        Inverted-bottleneck design: cheap depthwise convolutions carry the spatial work, so the parameters go
        into more channels and more blocks than a ResNet of the same size can afford.
        64x64 input -> stem stride 2 (32x32) -> 2 bottlenecks at 16x16 -> 2 bottlenecks at 8x8 -> 1x1 conv -> head.
        With res=96 a max-pool after the stem keeps the same feature-map sizes and the same parameters.
        Same module layout as the ablation models, so their weights load with strict=True.
    '''

    def __init__(self, dropout=DROPOUT, res=RES):
        assert res in (64, 96), "GhostNet supports res 64 or 96"
        stem = [conv_bn(3, 14, 3, 2)] + ([torch.nn.MaxPool2d(2)] if res == 96 else [])
        super().__init__(
            *stem,
            GhostBottleneck(14, 36, 16, 2),
            GhostBottleneck(16, 52, 16),
            GhostBottleneck(16, 52, 28, 2),
            GhostBottleneck(28, 72, 28),
            conv_bn(28, 44, 1),
            Head(44, dropout),
        )


class MiniResNet(torch.nn.Module):

    '''
        PREVIOUS ARCHITECTURE DESIGN: Mini-RESNET

        The idea is to spend a tiny parameter budget as efficiently as possible: keep enough channels from the start to avoid losing information,
        reuse the same width throughout, and use residual connections so the network trains easily and can be regularized with stochastic depth.
    '''

    def __init__(self,dropout=0.2,sd_prob=0.1,res=RES):
        super().__init__()

        # The stem brings any resolution down to 16x16, without adding parameters (stride + pooling)
        assert res in (16, 32, 64, 128), "unsupported resolution"
        stride = 2 if res >= 64 else 1
        size = res // stride
        downsample = []
        while size > 16:
            downsample.append(torch.nn.MaxPool2d(2))
            size //= 2
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,stride=stride,padding=1,bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            *downsample,
        )

        self.block1 = ResBlock(16,16,sd_prob/2) 

        self.block2 = ResBlock(16,16,sd_prob)

        self.pool = torch.nn.MaxPool2d(2)

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(16,len(CLASSES))
        )       

    def forward(self,x):
        x = self.stem(x)

        x = self.pool(self.block1(x))

        x = self.pool(self.block2(x))

        x = self.classifier(x)

        return x



class EMA:
    '''
        Exponential Moving Average of the weights.

        Keeps a shadow copy of the model whose weights are a running average of the trained weights:
            ema_w = decay * ema_w + (1 - decay) * w        (after every optimizer step)
        BatchNorm running stats are averaged the same way, so they stay consistent with the averaged weights.
        The shadow copy is a plain Net, so its state_dict has the same keys as the model and can be saved
        and loaded by challenge_test.py without changes. It is only used during training, so it does not
        count towards the parameter budget.
    '''

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        # Warm-up: early in training the average would be dominated by the random initialization,
        # so use a smaller decay at the start (1/11, 2/12, ... -> decay)
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        live = model.state_dict()
        for name, ema_t in self.model.state_dict().items():
            if ema_t.is_floating_point():
                ema_t.lerp_(live[name], 1 - d)    # ema = d*ema + (1-d)*live
            else:
                ema_t.copy_(live[name])           # e.g. BatchNorm num_batches_tracked


class IndexedDataset(torch.utils.data.Dataset):
    '''Wraps a dataset so each item is (x, y, idx); idx selects the row of the teacher logits.'''

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        return x, y, i


def build_teacher_logits(train_dir=TRAIN_DATA_DIR, out_file=TEACHER_FILE):
    '''Soft labels for distillation: {"logits": [N,10], "files": [N] paths} in ImageFolder order of train_dir.
    Frozen ImageNet backbone features (image + mirrored image averaged), then a logistic-regression head per
    fold; each image gets the logits of the head that did not see it. Runs once (a few minutes on a GPU).'''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = v2.Compose([
        v2.ToImage(), v2.Resize(TEACHER_RES), v2.CenterCrop(TEACHER_RES),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # ImageNet stats
    ])
    data = torchvision.datasets.ImageFolder(root=train_dir, transform=tf)
    n_classes = len(data.classes)
    with torch.random.fork_rng(devices=[]):          # own seed, leaves the training RNG untouched
        torch.manual_seed(TEACHER_SEED)
        weights = torchvision.models.get_model_weights(TEACHER_BACKBONE).DEFAULT
        backbone = torchvision.models.get_model(TEACHER_BACKBONE, weights=weights)
        backbone.classifier[-1] = torch.nn.Identity()     # keep the pooled features
        backbone = backbone.eval().to(device)
        # no loader workers: Docker's default 64 MB /dev/shm overflows with 224 px batches
        loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=0)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in tqdm.tqdm(loader, desc="teacher features"):
                x = x.to(device)
                feats.append(((backbone(x) + backbone(torch.flip(x, dims=[3]))) / 2).cpu())
                labels.append(y)
        X, y = torch.cat(feats), torch.cat(labels)
        del backbone

        # stratified folds: shuffle inside each class, deal the samples round-robin to the folds
        fold = torch.empty(len(y), dtype=torch.long)
        for c in range(n_classes):
            idx = (y == c).nonzero().squeeze(1)
            idx = idx[torch.randperm(len(idx))]
            fold[idx] = torch.arange(len(idx)) % TEACHER_FOLDS

        logits = torch.zeros(len(y), n_classes)
        for k in range(TEACHER_FOLDS):
            tr, te = fold != k, fold == k
            mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
            head = torch.nn.Linear(X.shape[1], n_classes).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=TEACHER_HEAD_WD)
            Xn, yt = ((X[tr] - mu) / sd).to(device), y[tr].to(device)
            for _ in range(TEACHER_HEAD_EPOCHS):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(head(Xn), yt).backward()
                opt.step()
            with torch.no_grad():
                logits[te] = head(((X[te] - mu) / sd).to(device)).cpu()
            print(f"teacher fold {k}: out-of-fold acc {(logits[te].argmax(1) == y[te]).float().mean():.4f}")
    print(f"Teacher out-of-fold train acc: {(logits.argmax(1) == y).float().mean():.4f}")
    saved = {"logits": logits, "files": [p for p, _ in data.samples]}
    torch.save(saved, out_file)
    return saved


def load_teacher_logits(files):
    '''Cached teacher logits for exactly these train files; rebuilt when missing or made for another train set.'''
    try:
        saved = torch.load(TEACHER_FILE, weights_only=True)
        if saved["files"] == files:
            return saved["logits"]
        print(f"{TEACHER_FILE} was made for a different train set: rebuilding it")
    except FileNotFoundError:
        print(f"{TEACHER_FILE} not found: building the teacher soft labels (runs once)")
    saved = build_teacher_logits()
    assert saved["files"] == files
    return saved["logits"]


def distillation_loss(student_logits, teacher_logits, y, T, alpha):
    '''alpha * CE(hard) + (1-alpha) * T^2 * KL(teacher || student) at temperature T. T^2 keeps the soft-term gradient scale.'''
    hard = torch.nn.functional.cross_entropy(student_logits, y)
    soft = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / T, dim=1),
        torch.nn.functional.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    return alpha * hard + (1 - alpha) * soft


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, use_gpu):
    '''Mean loss and accuracy over the whole dataset, in eval mode (no dropout / stochastic depth, BN running stats).'''
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in dataloader:
        if use_gpu and torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        y_pred = model(x)
        total_loss += loss_fn(y_pred, y).item() * len(x)   # loss_fn returns the batch mean
        correct += (y_pred.argmax(dim=1) == y).sum().item()
        n += len(x)
    return total_loss / n, correct / n


class Trainer(Test):
    
    def __init__(self, res=RES):
        super(Test, self).__init__()
        self.res = res


    def create_model(self):
        if ARCH == "ghost":
            return GhostNet(dropout=DROPOUT, res=self.res)
        return MiniResNet(sd_prob=0.1,dropout=0.2,res=self.res)
    
    
    def create_transform(self, inference=True):
        base, base_train, random_part, eval_part = make_transforms(self.res)
        return v2.Compose(base + eval_part if inference else base_train + random_part)
    

    def save_model(self, model, path="model.torch"):
        torch.save(model.state_dict(), path)

    def compute_stats(self):
        # COMPUTING MEAN AND STD
        plain = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.res),
            v2.CenterCrop(self.res),
            v2.ToDtype(torch.float32, scale=True),   # no Normalization
        ])
        data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=plain)
        loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
        X, _ = next(iter(loader))
        X = X.numpy()
        print("mean:", X.mean(axis=(0, 2, 3)).round(4))
        print("std:", X.std(axis=(0, 2, 3)).round(4))

    def dataloaders(self):
        base, base_train, random_part, eval_part = make_transforms(self.res)
        train_data = CachedDataset(TRAIN_DATA_DIR, base_train, random_part)
        train_hashes = {file_hash(p) for p, _ in train_data.samples}
        if DISTILL:
            self.teacher_logits = load_teacher_logits([p for p, _ in train_data.samples])
            train_data = IndexedDataset(train_data)
        val_data = CachedDataset(VAL_DATA_DIR, base, eval_part)
        # Val images that are byte-identical to a train image are left out, so they cannot drive epoch selection.
        # final_report() still measures the full val split.
        keep = [i for i, (p, _) in enumerate(val_data.samples) if file_hash(p) not in train_hashes]
        print(f"Val: {len(keep)} images used, {len(val_data) - len(keep)} duplicates of train images left out")
        val_data = torch.utils.data.Subset(val_data, keep)

        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
        val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

        return train_data,val_data,train_dataloader,val_dataloader


    def train_model(self, model,epoch_num,use_gpu,dataloader_train,dataloader_val):

        # Defining optimizer and loss
        decay, no_decay = [], []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": WEIGHT_DECAY},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=LR,
        )
        steps_per_epoch = len(dataloader_train)
        total_steps = epoch_num * steps_per_epoch
        warm = max(1, int(0.03 * total_steps))      # 3 % of training for warmup

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm            # LR rises from ~0 to LR
            p = (step - warm) / max(1, total_steps - warm)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p))   # LR falls to 1 % of LR

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss = torch.nn.CrossEntropyLoss()

        # move to GPU if GPU is available
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()

        # EMA shadow model (created after .cuda() so it lives on the same device)
        ema = EMA(model, EMA_DECAY) if EMA_DECAY is not None else None

        time_start = time.time()
        epochs = epoch_num

        best_val_acc = -1.0
        best_epoch = -1
        best_model_weights = copy.deepcopy(model.state_dict())

        for epoch in tqdm.trange(epochs):
            train_loss = 0.0
            train_correct = 0
            train_n = 0

            model.train()  # training mode (matters for layers such as dropout or batch norm)
            for batch in dataloader_train:
                if DISTILL:
                    x, y, idx = batch
                    t_logits = self.teacher_logits[idx]
                else:
                    x, y = batch
                # delete previous gradient calculations
                optimizer.zero_grad()
                # move to GPU if GPU is available
                if use_gpu and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                    if DISTILL:
                        t_logits = t_logits.cuda()
                if AUG_MODE == "gpu_strong":
                    x = gpu_augment(x, self.res)

                # predict class logits for each image in the batch
                y_pred = model(x)
                # compute the loss (mean over the batch); with distillation it mixes hard and teacher soft labels
                if DISTILL:
                    L = distillation_loss(y_pred, t_logits, y, TEMPERATURE, ALPHA)
                else:
                    L = loss(y_pred, y)
                # compute gradients
                L.backward()
                # change parameters of the network
                optimizer.step()
                scheduler.step()
                if ema is not None:
                    ema.update(model)

                # L is the mean over the batch, so weight it by the batch size
                train_loss += L.item() * len(x)
                train_correct += (y_pred.argmax(dim=1) == y).sum().item()
                train_n += len(x)

            # Divide by the number of IMAGES, not the number of batches
            train_loss /= train_n
            train_acc = train_correct / train_n

            # Val evaluation in order to obtain the best model, not the last one
            val_loss, val_acc = evaluate(model, dataloader_val, loss, use_gpu)
            if ema is not None:
                ema_val_loss, ema_val_acc = evaluate(ema.model, dataloader_val, loss, use_gpu)

            # Model selection: with EMA enabled we select (and save) the EMA weights
            selected_model = ema.model if ema is not None else model
            selected_acc = ema_val_acc if ema is not None else val_acc
            if selected_acc > best_val_acc:
                best_val_acc = selected_acc
                best_epoch = epoch
                best_model_weights = copy.deepcopy(selected_model.state_dict())

            # some output
            print(f"[{epoch:>2}] LR: {scheduler.get_last_lr()[0]:.6f}  "
                  f"Best val acc so far: {best_val_acc:.5f} (epoch {best_epoch})")
            # NOTE: train acc/loss are measured on AUGMENTED images in TRAIN mode (dropout, stochastic depth),
            # averaged while the weights change, so they are not comparable to challenge_test.py's "train"
            print(f"[{epoch:>2}] Train accuracy (augmented, train mode): {train_acc:.5f}")
            print(f"[{epoch:>2}] Train loss (augmented, train mode): {train_loss:.5f}")
            print(f"[{epoch:>2}] Validation accuracy: {val_acc:.5f}   loss: {val_loss:.5f}")
            if ema is not None:
                print(f"[{epoch:>2}] EMA validation accuracy: {ema_val_acc:.5f}   loss: {ema_val_loss:.5f}")
            time_past = time.time() - time_start
            time_per_epoch = time_past / (epoch+1)
            eta = (epochs-1-epoch) * time_per_epoch
            print(f"[{epoch:>2}] Time: {time_past:.1f}s, Time-per-Epoch: {time_per_epoch:.1f}s, eta: {eta:.1f}s")

        print(f"Last epoch val acc: live {val_acc:.5f}" + (f", EMA {ema_val_acc:.5f}" if ema is not None else ""))
        print(f"Best epoch: {best_epoch}/{epochs-1}, val acc WITHOUT duplicates: {best_val_acc:.5f}" + (" (EMA weights)" if ema is not None else ""))
        if SELECT == "last":
            print("Saving the last-epoch " + ("EMA " if ema is not None else "") + "weights (SELECT = 'last')")
            if ema is not None:
                model.load_state_dict(ema.model.state_dict())
        else:
            model.load_state_dict(best_model_weights)
        return model

    def final_report(self, model, use_gpu):
        '''Clean accuracy (eval transform, eval mode): the same numbers challenge_test.py reports.'''
        loss = torch.nn.CrossEntropyLoss()
        for split, root in [("train", TRAIN_DATA_DIR), ("val", VAL_DATA_DIR)]:
            data = torchvision.datasets.ImageFolder(root=root, transform=self.create_transform())
            loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            _, acc = evaluate(model, loader, loss, use_gpu)
            print(f"Final clean {split} accuracy (full split, as challenge_test.py): {acc:.3f}")
    
if __name__ == "__main__":
    torch.manual_seed(SEED)

    trainer = Trainer()
    model = trainer.create_model()

    # trainer.compute_stats()

    _,_,dataloader_train,dataloader_val = trainer.dataloaders()

    # print(f"Uno: {len(dataloader_train)}")
    # print(f"Dos: {len(dataloader_train.dataset)}")

    model = trainer.train_model(model=model,epoch_num=EPOCH_NUM,use_gpu=USE_GPU,dataloader_train=dataloader_train,dataloader_val=dataloader_val)

    num_par = sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()]).item()

    print(num_par)

    trainer.save_model(model)     # model.torch: the one challenge_test.py evaluates
    trainer.final_report(model, USE_GPU)
