import torch
import pytorch_lightning as pl
from torch import nn,optim
from torch.utils import data 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from lib import datasetloader, nets, utils
from pytorch_lightning import Trainer

trainer_model = utils.ModelTrainer('custom', 10, True, 'custom', True)
dataloader = datasetloader.Fashion_Mnist_DL()


train_dataset, val_dataset, test_dataset = dataloader.prepare_data()

train_dataloader = dataloader.create_dataloader(train_dataset, use_tpu=False, batch_size=8)
val_dataloader = dataloader.create_dataloader(val_dataset, use_tpu=False, batch_size=8)
test_dataloader = dataloader.create_dataloader(test_dataset, use_tpu=False, batch_size=8)


trainer = Trainer(gpus=1, max_epochs=5)

trainer.fit(trainer_model, train_dataloader, val_dataloader, test_dataloader)

