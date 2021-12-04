import pytorch_lightning as pl
from lib.nets import Custom_CNN, Custom_CNN1, PretainedModels_Finetuning
import torch.nn.functional as F
import torch

class ModelTrainer(pl.LightningModule):

    def __init__(self,model_name, num_classes, feature_extract,  model_class, use_pretrained =True):
        super(ModelTrainer, self).__init__()

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
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        loss = F.cross_entropy(preds, target)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #print(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        
    def test_step(self, batch, batch_idx):
        images, target = batch
        preds = self.base_class.forward(images)
        return {'test_loss': F.cross_entropy(preds, target)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.models.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]
            