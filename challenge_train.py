# -*- coding: utf-8 -*-
"""

    This is an example implementation of what you have to submit in order
    to test your model on the "test" split.

"""

from challenge_test import Test
import torch, torchvision, torchvision.transforms.v2 as v2
import numpy as np
import time
import tqdm
from torchvision.ops import StochasticDepth
import copy
import math




# VARIABLES DEFINITION
TRAIN_DATA_DIR = "dataset/train/"
VAL_DATA_DIR = "dataset/val/"
BATCH_SIZE = 128
KERNEL_SIZE = 3
EPOCH_NUM = 250
CLASSES = ["apple","avocado","banana","cherry","kiwi","mango","orange","pineapple","strawberry","watermelon"]
USE_GPU = True
LR = 0.001 # Default:0.001
LAMBDA_REG = 0.01
# EMA of the weights. None disables it. The EMA averages roughly the last 1/(1-decay) optimizer steps
# (with 18 steps/epoch, 0.99 ~ last 100 steps ~ 5-6 epochs).
EMA_DECAY = 0.99
# Calculated through commented code in datatrain
MEAN = [0.6492, 0.5511, 0.402 ]
STD = [0.2646, 0.2748, 0.3047]

train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(16),
    v2.CenterCrop(16),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomResizedCrop(16, scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),   # computed from training set
    # v2.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # optional, small patches
])

'''
train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(16),
    v2.CenterCrop(16), # Not all images are squared
    v2.RandomResizedCrop(16, scale=(0.6, 1.0), ratio=(0.85, 1.15), antialias=True),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
    v2.RandomHorizontalFlip(0.5),  
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
    v2.RandomErasing(p=0.5, scale=(0.02, 0.2), value=0),
])
'''

eval_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(16),
    v2.CenterCrop(16),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),
])

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


class Net(torch.nn.Module):

    '''
        NEW ARCHITECTURE DESIGN: Mini-RESNET

        The idea is to spend a tiny parameter budget as efficiently as possible: keep enough channels from the start to avoid losing information,
        reuse the same width throughout, and use residual connections so the network trains easily and can be regularized with stochastic depth.
    '''

    def __init__(self,dropout=0.2,sd_prob=0.1):
        super().__init__()

        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,3,padding=1,bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
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
    
    def __init__(self):
        super(Test, self).__init__()


    def create_model(self):
        return Net(sd_prob=0.1,dropout=0.2)
    
    
    def create_transform(self, inference=True):
        if inference:
            return eval_transform
        return train_transform
    

    def save_model(self, model):
        torch.save(model.state_dict(), "model.torch")

    def compute_stats(self):
        # COMPUTING MEAN AND STD
        plain = v2.Compose([
            v2.ToImage(),
            v2.Resize(16),
            v2.CenterCrop(16),
            v2.ToDtype(torch.float32, scale=True),   # no Normalization
        ])
        data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=plain)
        loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False)
        X, _ = next(iter(loader))
        X = X.numpy()
        print("mean:", X.mean(axis=(0, 2, 3)).round(4))
        print("std:", X.std(axis=(0, 2, 3)).round(4))

    def dataloaders(self):
        train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_DIR,transform=train_transform)
        val_data = torchvision.datasets.ImageFolder(root=VAL_DATA_DIR,transform=eval_transform)

        train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
        val_dataloader = torch.utils.data.DataLoader(val_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

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
                {"params": decay,    "weight_decay": 5e-2},
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
            for x, y in dataloader_train:
                # delete previous gradient calculations
                optimizer.zero_grad()
                # move to GPU if GPU is available
                if use_gpu and torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # predict class logits for each image in the batch
                y_pred = model(x)
                # compute the loss (mean over the batch)
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
        print(f"Best epoch: {best_epoch}/{epochs-1}, val acc: {best_val_acc:.5f}" + (" (EMA weights)" if ema is not None else ""))
        model.load_state_dict(best_model_weights)
        return model

    def final_report(self, model, use_gpu):
        '''Clean accuracy (eval transform, eval mode): the same numbers challenge_test.py reports.'''
        loss = torch.nn.CrossEntropyLoss()
        for split, root in [("train", TRAIN_DATA_DIR), ("val", VAL_DATA_DIR)]:
            data = torchvision.datasets.ImageFolder(root=root, transform=eval_transform)
            loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
            _, acc = evaluate(model, loader, loss, use_gpu)
            print(f"Final clean {split} accuracy: {acc:.3f}")
    
if __name__ == "__main__":
    trainer = Trainer() 
    model = trainer.create_model()

    # trainer.compute_stats()

    _,_,dataloader_train,dataloader_val = trainer.dataloaders()

    # print(f"Uno: {len(dataloader_train)}")
    # print(f"Dos: {len(dataloader_train.dataset)}")

    model = trainer.train_model(model=model,epoch_num=EPOCH_NUM,use_gpu=USE_GPU,dataloader_train=dataloader_train,dataloader_val=dataloader_val)

    num_par = sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()]).item()

    print(num_par)

    trainer.save_model(model)
    trainer.final_report(model, USE_GPU)