import torch
import pytorch_lightning as pl
from torch import nn, optim
from lib import datasetloader, nets, utils
import argparse
from time import sleep
from json import dumps
import cv2
from fastapi import FastAPI, HTTPException
import fastapi
import logging 
import os
import sys
import uvicorn
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path
from starlette.middleware.cors import CORSMiddleware
import asyncio
from torchvision import datasets,transforms
import numpy as np



class InferClassifier():

    def __init__(self, checkpoint_path, transform_type):
        self.description = "Inference class call to run the Inference on a given model"
        print("Model checkpoints loading")
        model = utils.ModelTrainer('custom1',10, False, 'custom1')
        self.model = model.base_class
        print(type(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']))
        old_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict']
        new_weights  = self.model.state_dict()
        print("======================================================================")
        print(old_weights.keys())
        print("======================================================================")
        print(new_weights.keys())
        i =0
        for k,_ in new_weights.items():
            print(k)
            new_weights[k] = old_weights['base_class.'+str(k)]
            print(new_weights[k].shape)
            i +=1
        self.model.load_state_dict(new_weights)
        #self.model = pl.LightningModule.load_from_checkpoint(checkpoint_path)
        print("Model checkpoint laoded successfully")
        self.model.eval().to(device='cpu')
        self.transform_type = transform_type

    def img_transformations(self, image):
        if self.transform_type == "custom_model":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28,28), interpolation=cv2.INTER_AREA)
        image_tensor = torch.from_numpy(image/255.).float()[None,None].to('cpu')
        print(image_tensor.shape)
        return image_tensor
    def infer_image(self, image):
        transformed_image = self.img_transformations(image)
        print("Before Prediction")
        predictions =  self.model(transformed_image)
        print("Predictions:{}".format(predictions))
        predictions = predictions.argmax()
        print("Formatted Prediction:{}".format(predictions))
        print("After Prediction")
        classes_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        prediction = classes_list[predictions]
        return prediction


app = fastapi.FastAPI()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
checkpoint_path = "./logs/default/version_3/checkpoints/epoch=13-step=13131.ckpt"
transform_type = "custom_model1"
inference_classifer = InferClassifier(checkpoint_path, transform_type)

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"])

@app.get("/")
def entry_page():
    return "Testing Rest-API implementation of the Image Classifier Inference class"

@contextmanager
def temporary_upload(fileobj):
    with TemporaryDirectory() as td:
        filename = fileobj.filename
        filepath = Path(td) / filename
        with filepath.open('wb') as f:
            loop = asyncio.new_event_loop()
            f.write(loop.run_until_complete(fileobj.read()))
            loop.stop()
            loop.close()
        yield filepath


@app.post("/infer_image/")
def infer_image(file: fastapi.UploadFile= fastapi.File(...)):
    with temporary_upload(file) as filepath:
        try:
            #print(filepath)
            image = cv2.imread(str(filepath))
            #print(image.shape)
            predictions = inference_classifer.infer_image(image)
            return predictions
        except Exception as ex:
            print("Exception :{}".format(ex))
            return 500

if __name__ == '__main__':
    uvicorn.run('inference_server:app', host = '0.0.0.0', port = 5005, proxy_headers =True, reload=True)




    

