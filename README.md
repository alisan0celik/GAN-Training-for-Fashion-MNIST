# Generative Adversarial Network (GAN) for Fashion MNIST

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for generating synthetic images resembling the Fashion MNIST dataset. GANs consist of two neural networks, a generator, and a discriminator, trained adversarially to generate realistic data.

### Prerequisites
Before running the code, ensure you have the following dependencies installed: PyTorch ,
torchvision,
matplotlib

### Dataset 

The Fashion MNIST dataset is used for training the GAN. It includes grayscale images of 10 different fashion categories. The dataset is loaded using PyTorch's torchvision library.

## Model Architecture

### Generator

The generator is a neural network with three fully connected layers, utilizing ReLU activation functions and a final tanh activation to produce generated images.

### Discriminator

The discriminator is also a neural network with three fully connected layers, using ReLU activation functions and a final sigmoid activation to classify images as real or fake.

### Model Initialization

The weights of the generator and discriminator are initialized using a custom weight initialization function. This helps in stabilizing the training process.

 ### Loss Function and Optimizers

 The Binary Cross Entropy (BCE) loss function is employed to train both the generator and discriminator. Adam optimizers are used for updating the model parameters.

 ### Learning Rate Scheduling

 Exponential learning rate scheduling is applied to both the generator and discriminator optimizers to enhance training stability and convergence.

 ### Training

 The GAN is trained over a specified number of epochs. In each epoch, the generator and discriminator are alternately trained on batches of real and generated images. The training process is visualized by printing the average generator and discriminator losses.

 ## Results

 Generated images are visualized every specified number of epochs to observe the progression of the generator. The generated images showcase the model's ability to produce synthetic data resembling the Fashion MNIST dataset.

 ## Author

 The code was created by Alişan Çelik
