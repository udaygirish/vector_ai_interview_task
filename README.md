## Branch dev/task1 - Explanation
Problem statement: Write a train script  for training an image classifier which on default mode can train with the Fashion MNIST data and also supports other datasets which are formatted to Pytorch Image Folder Dataloader format.


### Code Explanation -  Arguments and Sample run
Currently code supports three  modelling approaches:
- 2 Custom Built networks
- One Pretrained network class which inbuilt supports multiple supports training for ResNet18,Alexnet,VGG, Squeezenet, Densenet,Inception.

1. Sample run Train script:
    - python3 train.py -dp <dataset_path> -mn <model_name>  -mc <model_class> -n <num_classes> ..