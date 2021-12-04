import pytorch_lightning as pl
from torchmetrics.classification import average_precision
from lib.nets import Custom_CNN, Custom_CNN1, PretainedModels_Finetuning
import torch.nn.functional as F
import torch
from torchmetrics import Accuracy, AveragePrecision, AUROC, CohenKappa, F1, Precision, Recall

class ModelTrainer(pl.LightningModule):

    def __init__(self,model_name, num_classes, feature_extract,  model_class, use_pretrained =True):
        super(ModelTrainer, self).__init__()

        self.accuracy = Accuracy()
        self.average_precision = AveragePrecision(num_classes,  average=None)
        self.auroc = AUROC(num_classes)
        self.cohenkappa = CohenKappa(num_classes)
        self.f1_score = F1(num_classes)
        self.precision = Precision(average='macro', num_classes=num_classes)
        self.recall = Recall(average='macro', num_classes=num_classes)

        if model_class == "custom":
            self.base_class = Custom_CNN(num_classes)
        
        elif model_class == "custom1":
            self.base_class = Custom_CNN1(num_classes)

        elif model_class == "pretrained_finetune":
            self.base_class = PretainedModels_Finetuning(model_name, num_classes, feature_extract, use_pretrained=True)

        #self.forward = self.base_class.forward(x)


    def training_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        loss = F.cross_entropy(preds, target)
        accuracy = self.accuracy(preds,target)
        precision = self.precision(preds,target)
        recall = self.recall(preds,target)
        f1_score = self.f1_score(preds,target)
        tensorboard_logs = {'train_loss': loss,
                            'train_precision': precision,
                            'train_recall': recall,
                            'train_f1': f1_score}

        return {'loss': loss, 'log': tensorboard_logs, 'train_precision': precision,
                            'train_recall': recall,
                            'train_f1': f1_score}
    
    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        loss = F.cross_entropy(preds, target)
        accuracy = self.accuracy(preds,target)
        precision = self.precision(preds,target)
        recall = self.recall(preds,target)
        f1_score = self.f1_score(preds,target)
        return {'val_loss': loss, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1_score}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_precision = torch.stack([x['val_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['val_recall'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()

        #print(avg_loss)
        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_precision': avg_precision,
                            'avg_val_recall': avg_recall,
                            'avg_val_f1': avg_f1}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs,'avg_val_precision': avg_precision,
                            'avg_val_recall': avg_recall,
                            'avg_val_f1': avg_f1 }
        
    def test_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        accuracy = self.accuracy(preds,target)
        precision = self.precision(preds,target)
        recall = self.recall(preds,target)
        f1_score = self.f1_score(preds,target)
        return {'test_loss': F.cross_entropy(preds, target), 'test_precision': precision, 'test_recall': recall, 'test_f1': f1_score}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
        avg_recall = torch.stack([x['test_recall'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss, 'avg_test_precision': avg_precision, 'avg_test_recall': avg_recall, 'avg_test_f1': avg_f1}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs, 'avg_test_precision': avg_precision, 'avg_test_recall': avg_recall, 'avg_test_f1': avg_f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
            