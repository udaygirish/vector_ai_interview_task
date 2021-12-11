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




class InferClassifier():

    def __init__(self, checkpoint_path, transform_type):
        self.description = "Inference class call to run the Inference on a given model"
        self.model = pl.LightningModule.load_from_checkpoint(checkpoint_path)
        self.model.eval()
        self.transform_type = transform_type

    def img_transformations(self, image):
        if transform_type == "custom_model":
            image = 

    def infer_image(self, image):
        transformed_image = self.img_transformations(image)
        predictions =  self.model(transformed_image)
        return predictions


app = fastapi.FastAPI()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
checkpoint_path = ""
transform_type = ""
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
            image = cv2.imread(filepath)
            predictions = inference_classifer.infer_image(image)
            return predictions
        except Exception as ex:
            return {'Exception': ex}


    

