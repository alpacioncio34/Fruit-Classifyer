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



# VARIABLES DEFINITION
TRAIN_DATA_DIR = "dataset/train/"
VAL_DATA_DIR = "dataset/val/"
BATCH_SIZE = 64
KERNEL_SIZE = 3
EPOCH_NUM = 300
CLASSES = ["apple","avocado","banana","cherry","kiwi","mango","orange","pineaple","strawberry","watermelon"]
USE_GPU = True
LR = 0.001 # Default:0.001
LAMBDA_REG = 0.01
# Calculated through commented code in datatrain
MEAN = [0.6492, 0.5511, 0.402 ]
STD = [0.2646, 0.2748, 0.3047]

train_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(16),
    v2.CenterCrop(16),
    v2.RandomCrop(16, padding=2, padding_mode='reflect'),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=MEAN, std=STD),   # computed from training set
    v2.RandomErasing(p=0.25, scale=(0.02, 0.1)),  # optional, small patches
])

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

class Net(torch.nn.Module):

    '''
        CURRENT STATE OF THE NETWORK: UNACCEPTABLE:

        The model overfits even in the validation data (easier than the testing data) 40% val and 25% test acc.

        I propose using several techniques

        DONE: Image normalization

        DONE: Batch Normalization

        DONE: Applying transforms in order for the model to see different versions of each image in each epoch, a few randomized transformations

        DONE: Adding dropout at the end of the convolutional layers, might not work perfectly

        TODO: Keep trying weight decay parameters and epochs with higher decay

        TODO: Look for claude's proposed architecture

        After everything here is done and tested, I will report the networks performance increase.

        There are also other ideas to test, for example increasing the dataset by manually creating transformed images of the categories, but
        the main ones that are a must are the TODOs from above.
    '''

    def __init__(self,drop_prob = 0.0):
        super().__init__()

        # Here we define the layers, that will later be executed in forward
        self.drop_prob = drop_prob
        self.conv1 = torch.nn.Conv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=3,
            out_channels=2,
            padding="same")
        self.bn2 = torch.nn.BatchNorm2d(4)

        self.pool = torch.nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = torch.nn.Conv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=2,
            out_channels=4,
            padding="same")
        self.conv3 = torch.nn.Conv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=4,
            out_channels=8,
            padding="same"
        )
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=8,
            out_channels=16,
            padding="same"
        )
        self.bn4 = torch.nn.BatchNorm2d(16)
        self.conv5 = torch.nn.Conv2d(
            kernel_size=KERNEL_SIZE,
            in_channels=16,
            out_channels=32,
            padding="same"
        )
        self.bn5 = torch.nn.BatchNorm2d(32)
        self.dropout = torch.nn.Dropout(p=0.4)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            #PrintSize(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, len(CLASSES)),
        )
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.bn5(x)

        # Classifier layer
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

class Trainer(Test):
    
    def __init__(self):
        super(Test, self).__init__()


    def create_model(self):
        return Net()
    
    
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


    def train_model(self, model,epoch_num,use_gpu,dataloader_train):

        # Defining optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=5e-2)
        loss = torch.nn.CrossEntropyLoss()

        # move to GPU if GPU is available
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()

        model.reset_parameters()

        time_start = time.time()
        epochs = epoch_num

        for epoch in tqdm.trange(epochs):
            train_loss = 0
            train_correct = 0

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

                # L is the mean over the batch, so weight it by the batch size
                train_loss += L.item() * len(x)
                train_correct += (y_pred.argmax(dim=1) == y).sum().item()

            # some output
            print(f"[{epoch:>2}] Train loss: {train_loss:.5f}")
            time_past = time.time() - time_start
            time_per_epoch = time_past / (epoch+1)
            eta = (epochs-1-epoch) * time_per_epoch
            print(f"[{epoch:>2}] Time: {time_past:.1f}s, Time-per-Epoch: {time_per_epoch:.1f}s, eta: {eta:.1f}s")

        return model

if __name__ == "__main__":
    trainer = Trainer() 
    model = trainer.create_model()

    # trainer.compute_stats()

    _,_,dataloader_train,_ = trainer.dataloaders()

    # print(f"Uno: {len(dataloader_train)}")
    # print(f"Dos: {len(dataloader_train.dataset)}")

    model = trainer.train_model(model=model,epoch_num=EPOCH_NUM,use_gpu=USE_GPU,dataloader_train=dataloader_train)

    num_par = sum([torch.prod(torch.tensor(p.shape)) for p in model.parameters()]).item()

    print(num_par)

    trainer.save_model(model)