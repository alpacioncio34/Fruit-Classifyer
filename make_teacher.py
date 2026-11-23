# -*- coding: utf-8 -*-
"""
    Builds the soft labels used for knowledge distillation (run once, before challenge_train.py).

    Teacher = frozen ImageNet-pretrained backbone (high resolution input) + a linear head over 10 classes.
    To avoid a teacher that has "memorised" the training set, the logits for every training image come from a
    head that was NOT fitted on that image (K-fold out-of-fold predictions). The backbone is never trained.

    Output: teacher_logits.pt = {"logits": [N,10] float tensor, "files": [N] paths}
    N follows the ImageFolder sample order of TRAIN_DATA_DIR, so row i matches dataset sample i.

    NOTE: check the challenge rules before submitting with a pretrained teacher. The teacher is only used at
    training time and does not count towards the 10k parameter budget, but the rules may still forbid it.
    To use a different teacher, only the logits file matters: any [N,10] tensor in the same order works.
"""

import torch, torchvision, torchvision.transforms.v2 as v2
import tqdm

TRAIN_DATA_DIR = "dataset/train/"
OUT_FILE = "teacher_logits.pt"
BACKBONE = "convnext_tiny"      # torchvision model with ImageNet weights (downloaded on first use)
RES = 224
FOLDS = 5
HEAD_EPOCHS = 300
HEAD_WD = 1e-3
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_backbone():
    weights = torchvision.models.get_model_weights(BACKBONE).DEFAULT
    model = torchvision.models.get_model(BACKBONE, weights=weights)
    model.classifier[-1] = torch.nn.Identity()      # convnext: drop the final Linear, keep pooled features
    return model.eval().to(DEVICE)


@torch.no_grad()
def extract_features(model, data):
    loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False, num_workers=0)  # no workers: Docker's 64MB /dev/shm overflows with 224px batches
    feats, labels = [], []
    for x, y in tqdm.tqdm(loader, desc="features"):
        x = x.to(DEVICE)
        f = (model(x) + model(torch.flip(x, dims=[3]))) / 2     # average with the mirrored image
        feats.append(f.cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def fit_head(X, y, n_classes):
    '''Multinomial logistic regression on frozen features (features standardised with the fit-set stats).'''
    mu, sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
    head = torch.nn.Linear(X.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=HEAD_WD)
    Xn, y = ((X - mu) / sd).to(DEVICE), y.to(DEVICE)
    for _ in range(HEAD_EPOCHS):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(head(Xn), y).backward()
        opt.step()
    return lambda Z: head(((Z - mu) / sd).to(DEVICE)).detach().cpu()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    tf = v2.Compose([
        v2.ToImage(),
        v2.Resize(RES), v2.CenterCrop(RES),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),   # ImageNet stats
    ])
    data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=tf)
    n_classes = len(data.classes)

    X, y = extract_features(load_backbone(), data)

    # Stratified folds: shuffle inside each class, deal samples round-robin to the folds
    fold = torch.empty(len(y), dtype=torch.long)
    for c in range(n_classes):
        idx = (y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx))]
        fold[idx] = torch.arange(len(idx)) % FOLDS

    logits = torch.zeros(len(y), n_classes)
    for k in range(FOLDS):
        tr, te = fold != k, fold == k
        predict = fit_head(X[tr], y[tr], n_classes)
        logits[te] = predict(X[te])
        print(f"fold {k}: out-of-fold acc {(logits[te].argmax(1) == y[te]).float().mean():.4f}")

    print(f"Teacher out-of-fold train acc: {(logits.argmax(1) == y).float().mean():.4f}")
    torch.save({"logits": logits, "files": [p for p, _ in data.samples]}, OUT_FILE)
