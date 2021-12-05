import torch
import pytorch_lightning as pl
from torch import nn,optim 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torch.utils.data.sampler import SubsetRandomSampler
# import torch_xla.core.xla_model as xm --> TPU training support

class Fashion_Mnist_DL():

    def __init__(self):
        self.description = "Fashion MNIST Dataloader"

    def data_transforms(self, input_size, transform_type ='train'):
        if transform_type=='train':
            transforms_ret = transforms.Compose([
                                        transforms.RandomResizedCrop(256,scale=(0.8, 1.0),ratio=(0.75, 1.33)),
                                        transforms.RandomRotation(degrees=15),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                    ])

        elif transform_type == 'pretrained_finetune':
            transforms_ret = transform = transforms.Compose([transforms.Resize(input_size),
                                transforms.Lambda(lambda image:image.convert('RGB')),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        else:
            transforms_ret = transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
        return transforms_ret

    def prepare_data(self, input_size, transform_method):
        train_dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=self.data_transforms(input_size, transform_method))
        validation_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=self.data_transforms(input_size,transform_method))
        # For code validation purposes test set and val set are same
        test_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=self.data_transforms(input_size,transform_method))
        
        return train_dataset, validation_dataset, test_dataset


    def create_dataloader(self,dataset,use_tpu=False, batch_size=64):
        sampler = None
        # TPU support -= Instructions can be given later
        # if use_tpu:
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         dataset,
        #         num_replicas=xm.xrt_world_size(),
        #         rank=xm.get_ordinal(),
        #         shuffle=True
        #     )

        #     loader = torch.utils.data.DataLoader(
        #         dataset,
        #         sampler=sampler,
        #         batch_size=batch_size
        #     )
        # else:
        loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle = True
        )
        return loader

