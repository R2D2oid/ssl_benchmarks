# -*- coding: utf-8 -*-
"""
source: https://docs.lightly.ai

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
from lightly.models.modules import NNMemoryBankModule
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.modules.heads import ProjectionHead
from lightly.models.modules.heads import SwaVProjectionHead
from lightly.models.modules.heads import SwaVPrototypes
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.utils import BenchmarkModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop

from NoisyTinyImagenetDataset import NoisyTinyImagenet


parser = argparse.ArgumentParser(description='SSL Training')
parser.add_argument('--model_name', default='SimCLR', help='select one model from: [MoCo, SimCLR, SimSiam, BarlowTwinsModel, BYOL, SwAV]')
parser.add_argument('--user_name', default='srangrej', help='your username')
parser.add_argument('--data', default='/usr/local/data02/zahra/datasets/tiny-imagenet-200', help='path tinyimagenet dataset')

args = parser.parse_args()

user=args.user_name
model_names=args.model_name
data_dir=args.data

num_workers = 8
memory_bank_size = 4096

#logs_root_dir = os.path.join('/usr/local/extstore01/zahra/ssl_benchmarks/output')
logs_root_dir = os.path.join('/scratch/'+user+'/colab/SSL_tinyimg/')

# set max_epochs to 800 for long run (takes around 10h on a single V100)
max_epochs = 100
knn_k = 200
knn_t = 0.1
classes = 200

# benchmark
n_runs = 1 # optional, increase to create multiple runs and report mean + std
batch_sizes = [256]

# use a GPU if available
gpus = -1 if torch.cuda.is_available() else 0
distributed_backend = 'ddp' if torch.cuda.device_count() > 1 else None

# Adapted from our MoCo Tutorial on CIFAR-10
#
# Replace the path with the location of your CIFAR-10 dataset.
# We assume we have a train folder with subfolders
# for each class and .png images inside.
#
# You can download `CIFAR-10 in folders from kaggle 
# <https://www.kaggle.com/swaroopkml/cifar10-pngs-in-folders>`_.

# Use SimCLR augmentations, additionally, disable blur for cifar10
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=64, # 64 runs out of GPU memory on rocket
    gaussian_blur=0.,
)

# Multi crop augmentation for SwAV
#swav_collate_fn = lightly.data.SwaVCollateFunction(
#    crop_sizes=[32],
#    crop_counts=[2], # 2 crops @ 32x32px
#)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])


# load train/test data
dataset_train_ssl = NoisyTinyImagenet(data_dir, 
                                      split='train', 
                                      n_classes=200, 
                                      noise_rate=0.0, 
                                      noise_type='non',
                                      transform=None)

dataset_train_kNN = NoisyTinyImagenet(data_dir, 
                                      split='train', 
                                      n_classes=200, 
                                      noise_rate=0.0, 
                                      noise_type='non',
                                      transform=None)

dataset_test = NoisyTinyImagenet(data_dir, 
                                  split='val', 
                                  n_classes=200, 
                                  noise_rate=0.0, 
                                  noise_type='non',
                                  transform=None)


dataset_train_ssl = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_ssl) 
dataset_train_kNN = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_train_kNN)
dataset_test = lightly.data.LightlyDataset.from_torch_dataset(dataset=dataset_test)

dataset_train_kNN.transform=test_transforms
dataset_test.transform=test_transforms


def get_data_loaders(batch_size: int, multi_crops: bool = False):
    """Helper method to create dataloaders for ssl, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    dataloader_train_ssl = torch.utils.data.DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn if not multi_crops else swav_collate_fn,
        drop_last=True,
        num_workers=num_workers
    )

    dataloader_train_kNN = torch.utils.data.DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


class MocoModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)

        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18', num_splits=8)
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a moco model based on ResNet
        self.projection_head = MoCoProjectionHead(512, 512, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)
            
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = batch_shuffle(x1_)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = batch_unshuffle(x1_, shuffle)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(self.projection_head.parameters())
        optim = torch.optim.SGD(params, lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimCLRModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simclr model based on ResNet
        self.resnet_simclr = \
            lightly.models.SimCLR(self.backbone, num_ftrs=512)
        self.criterion = lightly.loss.NTXentLoss()
            
    def forward(self, x):
        self.resnet_simclr(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simclr(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simclr.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class SimSiamModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = lightly.models.ResNetGenerator('resnet-18')
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )
        # create a simsiam model based on ResNet
        self.resnet_simsiam = \
            lightly.models.SimSiam(self.backbone, num_ftrs=512)
        # replace the 3-layer projection head by a 2-layer projection head
        self.resnet_simsiam.projection_mlp = ProjectionHead([
            (
                self.resnet_simsiam.num_ftrs,
                self.resnet_simsiam.proj_hidden_dim,
                nn.BatchNorm1d(self.resnet_simsiam.proj_hidden_dim),
                nn.ReLU(inplace=True)
            ),
            (
                self.resnet_simsiam.proj_hidden_dim,
                self.resnet_simsiam.out_dim,
                nn.BatchNorm1d(self.resnet_simsiam.out_dim),
                None
            )
        ])
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
            
    def forward(self, x):
        self.resnet_simsiam(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_simsiam(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_simsiam.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BarlowTwinsModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        last_conv_channels = list(resnet.children())[-1].in_features
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
        )
        # create a barlow twins model based on ResNet
        self.resnet_barlowtwins = \
            lightly.models.BarlowTwins(
                self.backbone, 
                num_ftrs=512,
                proj_hidden_dim=2048,
                out_dim=2048,
            )
        # replace the 3-layer projection head by a 2-layer projection head
        self.resnet_barlowtwins.projection_mlp = ProjectionHead([
            (
                self.resnet_barlowtwins.num_ftrs,
                self.resnet_barlowtwins.proj_hidden_dim,
                nn.BatchNorm1d(self.resnet_barlowtwins.proj_hidden_dim),
                nn.ReLU(inplace=True)
            ),
            (
                self.resnet_barlowtwins.proj_hidden_dim,
                self.resnet_barlowtwins.out_dim,
                None,
                None
            )
        ])
        self.criterion = lightly.loss.BarlowTwinsLoss()

    def forward(self, x):
        self.resnet_barlowtwins(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        x0, x1 = self.resnet_barlowtwins(x0, x1)
        loss = self.criterion(x0, x1)
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_barlowtwins.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class BYOLModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        # create a byol model based on ResNet
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256,1024,256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            x0_ = self.projection_head(x0_)
            x0_ = self.prediction_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            return x0_, x1_

        p0, z1 = step(x0, x1)
        p1, z0 = step(x1, x0)
        
        loss = self.criterion((z0, p0), (z1, p1))
        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) \
            + list(self.projection_head.parameters()) \
            + list(self.prediction_head.parameters())
        optim = torch.optim.SGD(params, lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class SwaVModel(BenchmarkModule):
    def __init__(self, dataloader_kNN, num_classes):
        super().__init__(dataloader_kNN, num_classes)
        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1)
        )

        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512) # use 512 prototypes

        self.criterion = lightly.loss.SwaVLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):

        # normalize the prototypes so they are on the unit sphere
        lightly.models.utils.normalize_weight(
            self.prototypes.layers.weight
        )

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(
            high_resolution_features,
            low_resolution_features
        )

        self.log('train_loss_ssl', loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

models = {'MoCo':[MocoModel], 'SimCLR':[SimCLRModel], 'SimSiam':[SimSiamModel], 'BarlowTwinsModel':[BarlowTwinsModel], 'BYOL':[BYOLModel], 'SwAV':[SwaVModel]}
bench_results = []
gpu_memory_usage = []

# loop through configurations and train models
for batch_size in batch_sizes:
    for model_name, BenchmarkModel in zip([model_names], models[model_names]):
        runs = []
        for seed in range(n_runs):
            SAVEPATH=logs_root_dir+model_name
            pl.seed_everything(seed)
            dataloader_train_ssl, dataloader_train_kNN, dataloader_test = get_data_loaders(batch_size)
            benchmark_model = BenchmarkModel(dataloader_train_kNN, classes)

            logger_TBD = TensorBoardLogger(logs_root_dir, version=model_name, name='tb_logs')
            logger_CSV = CSVLogger(logs_root_dir, name=model_name)

            checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=SAVEPATH, filename=model_name+'_{epoch}', save_last=False, save_top_k=-1)
            trainer = pl.Trainer(max_epochs=max_epochs, 
                                gpus=gpus,
                                progress_bar_refresh_rate=0,
                                distributed_backend=distributed_backend,
                                default_root_dir=SAVEPATH,
                                logger=[logger_TBD, logger_CSV],
                                callbacks=[checkpoint_callback])
            trainer.fit(
                benchmark_model,
                train_dataloader=dataloader_train_ssl,
                val_dataloaders=dataloader_test
            )
            gpu_memory_usage.append(torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
            runs.append(benchmark_model.max_accuracy)

            # delete model and trainer + free up cuda memory
            del benchmark_model
            del trainer
            torch.cuda.empty_cache()
        bench_results.append(runs)

for result, model, gpu_usage in zip(bench_results, model_names, gpu_memory_usage):
    result_np = np.array(result)
    print(result_np.shape)
    mean = result_np.mean()
    std = result_np.std()
    with open(logs_root_dir+model_name+'/max_accuracy', 'a+') as f:
        f.write(f'{model}: {mean:.3f} +- {std:.3f}, GPU used: {gpu_usage / (1024.0**3):.1f} GByte')
