from time import sleep
from json  import dumps


import shutil
from fastapi import FastAPI, HTTPException
import os
import sys
import logging
import fastapi
import pydantic
from typing import Optional, List, Dict, Any
import uvicorn

from contextlib import contextmanager
from lib.google_pubsub import GcpPubSub
from lib.apache_kafka import ApacheKafka



## Initiaite GCP Pub sub and Apache Kafka
gcp_pub_sub = GcpPubSub()
apache_kafka = ApacheKafka()
###

## API Endpoints Declaration 
#CURRENTLY SUPPORTS APACHE KAFKA ONLY AS THE GCP PROJECT IS NOT THERE 
# YOU CAN USE THE CLASS LEVEL IMPLEMENTATION OF GCP PUBSUB IN FUTURE

app = fastapi.FastAPI()
logging.basicConfig(stream= sys.stdout, level = logging.DEBUG)

@app.get("/"):
def entry_page():
    return "Welcome to the base page of Google PubSub and Apache Kafka Publisher and Receiver app,\
            Please visit /docs to test the api. Two method calls are Publish and Receiver messages."


@app.post("/publish_message")
def publish_message(key:str, topic_name:str, value:str):
    # Connect to the producer
    apache_kafka.connect_kafka_producer()

    # Publish the message
    out_response = apache_kafka.publish_message(topic_name, key, value)
    if out_response == 0:
        return "Publishing Message failed. Please check Sys logs and data sanity"
    else:
        return "Message Published Successfully"

    
@app.post("/receive_message")
def receive_message(topic_name:str, auto_offset:str = 'earliest', timeout_ms:str = 100):
    # Consume non committed pending messages for futher processing
    received_mesages = apache_kafka.receive_message(topic_name, auto_offset, timeout_ms)
    return received_mesages


