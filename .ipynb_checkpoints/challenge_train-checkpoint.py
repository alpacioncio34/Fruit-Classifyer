# -*- coding: utf-8 -*-
"""

    This is an example implementation of what you have to submit in order
    to test your model on the "test" split.

"""

from challenge_test import Test
import torch, torchvision, torchvision.transforms.v2 as v2

class Trainer(Test):
    
    def __init__(self):
        super(Test, self).__init__()
    
    def create_model(self):
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*16**2, 10)
        )
    
    def create_transform(self, inference=True):
        return torchvision.transforms.Compose([
            v2.CenterCrop(16),
            torchvision.transforms.ToTensor(),
            v2.Normalize(
               mean = (0.5, 0.5, 0.5),
               std = (1.0, 1.0, 1.0)
            )
        ])
    
    def save_model(self, model):
        torch.save(model.state_dict(), "model.torch")
    
if __name__ == "__main__":
    trainer = Trainer()
    # create dummy weights for the test evaluation
    trainer.save_model(trainer.create_model())