# -*- coding: utf-8 -*-
"""

Note that this benchmark also supports a multi-GPU setup. If you run it on
a system with multiple GPUs make sure that you kill all the processes when
killing the application. Due to the way we setup this benchmark the distributed
processes might continue the benchmark if one of the nodes is killed.
If you know how to fix this don't hesitate to create an issue or PR :)


Code to reproduce the benchmark results:

| Model   | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|---------|--------|------------|---------------|----------------|
| MoCo    |  200   | 128        | 0.83          | 2.1 GBytes     |
| SimCLR  |  200   | 128        | 0.78          | 2.0 GBytes     |
| SimSiam |  200   | 128        | 0.73          | 3.0 GBytes     |
| MoCo    |  200   | 512        | 0.85          | 7.4 GBytes     |
| SimCLR  |  200   | 512        | 0.83          | 7.8 GBytes     |
| SimSiam |  200   | 512        | 0.81          | 7.0 GBytes     |
| MoCo    |  800   | 512        | 0.90          | 7.2 GBytes     |
| SimCLR  |  800   | 512        | 0.89          | 7.7 GBytes     |
| SimSiam |  800   | 512        | 0.91          | 6.9 GBytes     |

"""
import copy
import os

import lightly
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms
from all_models import MocoModel, SimCLRModel, SimSiamModel, BarlowTwinsModel, BYOLModel, SwaVModel
from NoisyCIFAR10Dataset import NoisyCIFAR10

parser = argparse.ArgumentParser(description='SSL Training')
parser.add_argument('--model_name', default='SimCLR', help='select one model from: [MoCo, SimCLR, SimSiam, BarlowTwinsModel, BYOL, SwAV]')
parser.add_argument('--user_name', default='srangrej', help='your username')
parser.add_argument('--noise_type', default='asym', help='asym or sym')
parser.add_argument('--noise_rate', default=0.0, type=float, help='from 0.0, 0.1, 0.2, ... , 0.9')
args = parser.parse_args()

user=args.user_name
model_name=args.model_name
noise_type=args.noise_type
noise_rate=args.noise_rate

num_workers = 8

logs_root_dir = os.path.join('/scratch/'+user+'/colab/SSL_50/')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 100
classes = 10

# benchmark
batch_size = 512

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0

# dataset
# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

train_base_dataset = NoisyCIFAR10(root='/home/'+user+'/projects/def-jjclark/shared_data/cifar10/data', train=True, download=False, noise_type = noise_type, noise_rate=noise_rate, split_ratio=0.5, transform=train_classifier_transforms)

dataset_train_classifier = lightly.data.LightlyDataset.from_torch_dataset(train_base_dataset)

# dataloader
dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

# dataset
test_base_dataset = NoisyCIFAR10(root='/home/'+user+'/projects/def-jjclark/shared_data/cifar10/data', train=False, download=False, noise_type = noise_type, noise_rate = 0.0, split_ratio=0.0, transform=test_transforms)

dataset_test = lightly.data.LightlyDataset.from_torch_dataset(test_base_dataset)

# dataloader
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


class Classifier(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        # create a moco based on ResNet
        self.model = model

        # freeze the layers of moco
        for p in self.model.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Linear(512, 10)

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.model.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=30.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


models = {'MoCo':MocoModel, 'SimCLR':SimCLRModel, 'SimSiam':SimSiamModel, 'BarlowTwinsModel':BarlowTwinsModel, 'BYOL':BYOLModel, 'SwAV':SwaVModel}
bench_results = []
gpu_memory_usage = []

model = models[model_name](None, classes)
model.load_state_dict(torch.load('/scratch/'+user+'/colab/SSL_50/'+model_name+'/'+model_name+'_epoch=199.ckpt')['state_dict'])
model.eval()

model_name = model_name+'_'+noise_type+'_'+str(noise_rate)
SAVEPATH=logs_root_dir+model_name

logger_TBD = TensorBoardLogger(logs_root_dir, version=model_name, name='tb_logs')
logger_CSV = CSVLogger(logs_root_dir, name=model_name)
checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=SAVEPATH, filename=model_name+'_{epoch}', save_last=False, save_top_k=-1)

pl.seed_everything(seed=1)

classifier = Classifier(model)
trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus,
                     progress_bar_refresh_rate=0,
                     default_root_dir=SAVEPATH,
                     logger=[logger_TBD, logger_CSV],
                     callbacks=[checkpoint_callback])
trainer.fit(
    classifier,
    dataloader_train_classifier,
    dataloader_test
)
