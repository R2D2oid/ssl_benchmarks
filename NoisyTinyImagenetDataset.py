import torch, os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

class NoisyTinyImagenet(Dataset):
    """Tiny-imagenet-200 dataste"""

    def __init__(self, data_dir, 
                 split='train', 
                 transform=None, 
                 n_classes=200,
                 noise_rate=0.0,
                 noise_type='sym'):
        """
        Args:
            data_dir (string): Path to the images.
            split (string): train/val/test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform #transforms.ToTensor() if transform is None else transform
        self.split = split
        self.ds = datasets.ImageFolder(os.path.join(data_dir, split), transform)
        
        self.data = [x[0] for x in self.ds]
        self.targets = [x[1] for x in self.ds]

        # add noise 
        self.noise_rate = noise_rate
        self.noise_type = noise_type

        random.seed(42)
        np.random.seed(42)
        
        if noise_rate <= 0:
            return
        
        # num samples in dataset
        n_samples = self.__len__()
        
        # num noisy samples to generate
        n_noisy_per_class = int(noise_rate * n_samples / n_classes)
        
        # for each class add noise to noise_rate percentage of its samples 
        for c in range(n_classes):
            indeces = np.where(np.array(self.targets) == c)[0]
            
            noisy_samples_idx = np.random.choice(indeces, n_noisy_per_class, replace=False)            
            
            if noise_type == 'sym':
                # list of alternative class ids to choose from as a noisy target; excludes original class id
                class_ids = [i for i in range(n_classes) if i!=c]    

                for idx in noisy_samples_idx:
                    # pick a new class from the remaining 9 classes at random as noisy class for this sample
                    new_target = random.choice(class_ids)
                    self.targets[idx] = new_target 
            elif noise_type == 'asym':
                for idx in noisy_samples_idx:
                    # use current_class+1 as the noisy class with prob noise_rate
                    current_class = self.targets[idx]
                    # np.random.seed(42)
                    self.targets[idx] = np.random.choice([current_class, (current_class+1)%n_classes], p=[1-noise_rate, noise_rate])
            else:
                raise ValueError(f'Undefined noise_type: {noise_type}!') 
                
    def __len__(self):
        assert len(self.data) == len(self.ds)
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        if self.transform:
            image = self.transform(image)
            
        return image, target
    
# # Usage examples
# from NoisyTinyImagenetDataset import NoisyTinyImagenet
# import torch
# from torch.utils.data import DataLoader

# data_dir = '/usr/local/data02/zahra/datasets/tiny-imagenet-200'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# tiny_ds_val=NoisyTinyImagenet(data_dir, split='val', n_classes=200, noise_rate=0.0, noise_type='non')
# tiny_ds_train_0=NoisyTinyImagenet(data_dir, split='train', n_classes=200, noise_rate=0.0, noise_type='non')
# tiny_ds_train_sym_1=NoisyTinyImagenet(data_dir, split='train', n_classes=200, noise_rate=0.1, noise_type='sym')
# tiny_ds_train_asym_1=NoisyTinyImagenet(data_dir, split='train', n_classes=200, noise_rate=0.1, noise_type='asym')

# dl = DataLoader(tiny_ds_train_sym_1, batch_size=256, shuffle=False, num_workers=4)

# for img,tgt in dl:
#     img.to(device)
#     tgt.to(device)
    
#     print(img.shape)
#     print(tgt.shape)
#     print(tgt)
#     break
