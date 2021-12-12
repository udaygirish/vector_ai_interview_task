# Vector AI Interview Take Home Task
Task Based Screening by Vector AI for Machine Learning Engineer Role.

1. There are currently three branches in the repo for the 3 different tasks mentioned in the Assessment.
2. The three branches are dev/task1, dev/task2, dev/task3.



## Branch dev/task1 - Explanation
Problem statement: Write a train script  for training an image classifier which on default mode can train with the Fashion MNIST data and also supports other datasets which are formatted to Pytorch Image Folder Dataloader format.


### Code Explanation -  Arguments and Sample run
Currently code supports three  modelling approaches:
- 2 Custom Built networks
- One Pretrained network class which inbuilt supports multiple supports training for ResNet18,Alexnet,VGG, Squeezenet, Densenet,Inception.

1. Sample run Train script:
    - python3 train.py -dp <dataset_path> -mn <model_name>  -mc <model_class> -n <num_classes> ..



## Branch dev/task2 - Explanation

Problem statement: Create a library which can publish and subscribe to messages using Google Cloud Pub Sub and Apache Kafka

Current code support both PubSub and Apache Kafka Implementations
But the test api is implemented only on Apache Kafka (Python lib)

1. Sample run API:
    -python3 main.py 
    - Two Endpoints
        - /publish_message --> Publish messages using Kafka Producer
        - /receive_message --> Subscribe to messages with timeout using Kafka Consumer


## Branch dev/task3  - Explanation

Problem Statement: Create a Application which can ingest continous messages from a fake Kafka Producer and run the inference on each of the image bytes in the message and save the result to a database/cli print.

1. Sample run API:
    - python3 app.py
    -  This app is still in Development

