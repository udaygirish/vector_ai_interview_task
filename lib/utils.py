import pytorch_lightning as pl
from lib.nets import Custom_CNN, Custom_CNN1, PretainedModels_Finetuning
import torch.nn.functional as F
import torch
from torchmetrics import (
    Accuracy,
    AveragePrecision,
    AUROC,
    CohenKappa,
    F1,
    Precision,
    Recall,
)


class ModelTrainer(pl.LightningModule):
    
    def __init__(
        self, model_name, num_classes, feature_extract, model_class, use_pretrained=True
    ):
        super(ModelTrainer, self).__init__()

        if model_class == "custom":
            self.base_class = Custom_CNN(num_classes)

        elif model_class == "custom1":
            self.base_class = Custom_CNN1(num_classes)

        elif model_class == "pretrained_finetune":
            self.base_class = PretainedModels_Finetuning(
                model_name, num_classes, feature_extract, use_pretrained=True
            )

        # self.forward = self.base_class.forward(x)
        self.accuracy = Accuracy()
        # self.average_precision = AveragePrecision(num_classes = num_classes,  average=None)
        # self.auroc = AUROC(num_classes=num_classes)
        # self.cohenkappa = CohenKappa(num_classess = num_classes)
        # self.f1_score = F1()
        # self.precision = Precision(average='micro')
        # self.recall = Recall(average='micro')
        self.model_class = model_class
        self.model_name = model_name
        self.feature_extract = feature_extract
        self.lr = 1e-5

    def training_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        loss = F.cross_entropy(preds, target)
        accuracy = self.accuracy(preds, target)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"loss": loss, "train_accuracy": accuracy}

    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        loss = F.cross_entropy(preds, target)
        accuracy = self.accuracy(preds, target)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": loss, "val_accuracy": accuracy}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["val_accuracy"] for x in outputs]).mean()

        # print(avg_loss)
        self.log(
            "avg_val_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "avg_val_accuracy",
            avg_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "avg_val_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
        }

    def test_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        accuracy = self.accuracy(preds, target)
        loss = F.cross_entropy(preds, target)
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_accuracy",
            accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"test_loss": loss, "test_accuracy": accuracy}

    def test_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x["test_accuracy"] for x in outputs]).mean()

        self.log(
            "avg_test_loss",
            avg_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "avg_test_accuracy",
            avg_accuracy,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {
            "avg_test_loss": avg_loss,
            "avg_accuracy": avg_accuracy,
        }

    def configure_optimizers(self):
        if self.model_class == "pretrained_finetune":
            model_ft = self.base_class.model_ft
            params_to_update = model_ft.parameters()
            print("Params to learn:")
            if self.feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\n", name)
            else:
                for name, param in model_ft.named_parameters():
                    if param.requires_grad == True:
                        print("\n", name)

            optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.base_class.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
