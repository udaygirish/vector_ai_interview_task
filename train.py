import torch
import pytorch_lightning as pl
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from lib import datasetloader, nets, utils
from pytorch_lightning import Trainer
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dp",
        "--dataset_path",
        required=False,
        default=None,
        help="Ensure this folder is kept in Data folder of this repo and only give the folder name",
    )
    parser.add_argument(
        "-tm",
        "--transform_method",
        required=False,
        default="custom",
        help="Arg to specify the transformations on the dataset",
    )
    parser.add_argument(
        "-mn",
        "--model_name",
        required=False,
        default="custom",
        help="Specify the model name as currently code supports multi model training",
    )
    parser.add_argument(
        "-mc",
        "--model_class",
        required=False,
        default="custom",
        help="To specify whether to use custom models or Finetuning on Pretrained models",
    )
    parser.add_argument(
        "-n",
        "--nc",
        required=False,
        default=10,
        help="Number of classes - Output classes",
    )
    parser.add_argument(
        "-mp",
        "--model_output",
        required=False,
        default="last_trained_model.pth",
        help="To specify the model output path to save state dict to .pth file",
    )
    parser.add_argument(
        "-fe",
        "--feature_extract",
        required=False,
        default=True,
        help="To specify whether to fine tune the pretrained model or trian from scratch",
    )
    parser.add_argument(
        "-bs",
        "--batchsize",
        required=False,
        default=32,
        help="To specify the batch size for training",
    )
    parser.add_argument(
        "-me",
        "--max_epoch",
        required=False,
        default=20,
        help="Maximum Epoch to be trained upon",
    )

    args = parser.parse_args()
    trainer_model = utils.ModelTrainer(
        args.model_name,
        args.nc,
        args.feature_extract,
        args.model_class,
        True,
    )
    transform_method = args.transform_method
    dataset_path = args.dataset_path

    if transform_method == "pretrained_finetune":
        input_size_int = trainer_model.base_class.input_size
        input_size = (input_size_int, input_size_int)
    else:
        input_size = (224, 224)

    if dataset_path == "fashion_mnist" or dataset_path == None:
        dataloader = datasetloader.Fashion_Mnist_DL()
    else:
        dataloader = datasetloader.Custom_Dataset(dataset_path)

    train_dataset, val_dataset, test_dataset = dataloader.prepare_data(
        input_size, transform_method
    )

    train_dataloader = dataloader.create_dataloader(
        train_dataset, use_tpu=False, batch_size=args.batchsize
    )
    val_dataloader = dataloader.create_dataloader(
        val_dataset, use_tpu=False, batch_size=args.batchsize
    )
    # test_dataloader = dataloader.create_dataloader(test_dataset, use_tpu=False, batch_size=32)

    # Tensorboard loggers
    tb_logger = pl.loggers.TensorBoardLogger("logs/")

    # MULTI GPU AND PARALLEL TRAINING USING DDP WITH PYTORCH LIGHTNING YET TO BE IMPLEMENTED
    trainer = Trainer(gpus=0, max_epochs=int(args.max_epoch), logger=tb_logger)

    trainer.fit(trainer_model, train_dataloader, val_dataloader)

    torch.save(trainer.model.state_dict(), args.model_output)


if __name__ == "__main__":
    main()
