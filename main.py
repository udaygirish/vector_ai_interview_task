import torch
import pytorch_lightning as pl
from torch import nn,optim
from torch.utils import data 
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from lib import datasetloader, nets, utils
from pytorch_lightning import Trainer

trainer_model = utils.ModelTrainer('custom', 10, False, 'custom', True)
transform_method = 'custom'
if transform_method == 'pretrained_finetune':
    input_size_int = trainer_model.base_class.input_size
    input_size = (input_size_int, input_size_int)
else:
    input_size = (224,224)
dataloader = datasetloader.Fashion_Mnist_DL()


train_dataset, val_dataset, test_dataset = dataloader.prepare_data(input_size, transform_method)

train_dataloader = dataloader.create_dataloader(train_dataset, use_tpu=False, batch_size=64)
val_dataloader = dataloader.create_dataloader(val_dataset, use_tpu=False, batch_size=32)
# test_dataloader = dataloader.create_dataloader(test_dataset, use_tpu=False, batch_size=32)

# Tensorboard loggers
tb_logger = pl.loggers.TensorBoardLogger("logs/")
trainer = Trainer(gpus=1, max_epochs=15, logger=tb_logger)

trainer.fit(trainer_model, train_dataloader, val_dataloader)

torch.save(trainer.model.state_dict(), 'model_custom.pth')