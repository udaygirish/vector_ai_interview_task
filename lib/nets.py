import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torchvision import models

class Custom_CNN(nn.Module):
    
    def __init__(self, num_classes:int):
        super(Custom_CNN, self).__init__()

        self.convlayer1 = nn.Sequential(nn.Conv2d(1,32,3,padding= 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.convlayer2 = nn.Sequential(nn.Conv2d(32,64,3), nn.BatchNorm2d(64),nn.ReLU(), nn.MaxPool2d(2))

        self.fc1 = nn.Linear(64*6*6, 600)
        self.dropout = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(600,120)
        self.fc3 = nn.Linear(120,num_classes)

    def forward(self,x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = x.view(-1, 64*6*6)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x_out = F.log_softmax(x,dim=1)

        return x_out

class Custom_CNN1(nn.Module):
    def __init__(self, num_classes:int):
        super(Custom_CNN1, self).__init__()

        self.convlayer1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding= 1), nn.ReLU(), nn.BatchNorm2d(32))
        self.convlayer2 = nn.Sequential(nn.Conv2d(32,64,3,padding =1), nn.ReLU(), nn.BatchNorm2d(64))
        self.convlayer3 = nn.Sequential(nn.Conv2d(64,64,3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(kernel_size=2, stride=1))
        self.dropout1 = nn.Dropout2d(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(64*6*6, 512), nn.ReLU())
        self.dropout2 = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(512,num_classes)

    def forward(self,x):
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        x = self.convlayer3(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        x_out = F.log_softmax(x)
        return x_out



class PretainedModels_Finetuning():

    def __init__(self,model_name, num_classes, feature_extract, use_pretrained =True):
        ## Method to call pretrained models for Finetuning to get base results
        self.description = "Class for PretrainedModels_finetuning"
        self.initialize_model(model_name, num_classes, feature_extract, use_pretrained =True)

        

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self,model_name, num_classes, feature_extract, use_pretrained =True):

        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs,num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        self.model_ft = model_ft
        self.input_size =  input_size


    def forward(self,x):
        output = self.models(x)
        return output








