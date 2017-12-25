# Image-Classification
Convolutional Neural Networks for classifying images

Programming Language: Python

Datasets used:
Cifar10 and Cifar100

An eight layered Convolutional Neural network along with two fully connected layers were trained on 50000 images with 15000 images used for validation and another 500 for testing. Different optimizers like adam, sgd, rmsprop were implemented to see how each one works with the given data. Softmax loss was used for classifying images to the ten categories available.
The number of feature maps were increased for the cifar100 dataset for extracting more sparse representation of images.

Accuracy:
Cifar10: 
Training Accuracy: 99%
Validation accuracy: 87%
Test Accuracy: 86%

Cifar100:
Training Accuracy: 99%
Validation accuracy: 62%
Test Accuracy: 60%
