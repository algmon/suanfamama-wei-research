# MamaNet1 Garment Classification Training
"""
This module is developed by Suanfamama.

Authors:
- Wei Jiang (wei@suanfamama.com)
- Mama Xiao (mama.xiao@suanfamama.com)

Reference:
- Training with PyTorch
- https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=nn%20crossentropyloss
"""

import torch
import torchvision
import torchvision.transforms as transforms

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

# Create data loaders for our datasets; shuffle for training, not for validation
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=4, shuffle=False)

# Class labels
classes = ('T-shirt', 'Trouser裤子', 'Pullover套衫', 'Dress裙子', 'Coat大衣',
        'Sandal拖鞋', 'Shirt衬衫', 'Sneaker运动鞋', 'Bag包', 'Ankle Boot靴子')

# Report split sizes
print('Training set has {} instances'.format(len(training_set)))
print('Validation set has {} instances'.format(len(validation_set)))

import matplotlib.pyplot as plt
import numpy as np

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(training_loader)
images, labels = next(dataiter)

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)
print('  '.join(classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F

# PyTorch models inherit from torch.nn.Module
class MamaNet1(nn.Module):
    def __init__(self):
        super(MamaNet1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MamaNet1()

loss_fn = torch.nn.CrossEntropyLoss()

# NB: Loss functions expect data in batches, so we're creating batches of 4
# Represents the model's confidence in each of the 10 classes for a given input
dummy_outputs = torch.rand(4, 10)
# Represents the correct class among the 10 being tested
dummy_labels = torch.tensor([1, 5, 3, 7])

print(dummy_outputs)
print(dummy_labels)

loss = loss_fn(dummy_outputs, dummy_labels)
print('Total loss for this batch: {}'.format(loss.item()))

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1

# Save the model checkpoint
import os
model_save_dir = "./"  # Directory to save the models
model_path = os.path.join(model_save_dir, f'MamaNet1_Garment_Classification.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

'''
Code Review

Reviewer: Suanfamama internal

Overall:

The script is a good for training a image based classification model using FashionMNIST and PyTorch. It demonstrates several key aspects of training a neural network.

Key Aspects:

* Clear Structure: The code is well-organized with clear sections for data loading, model definition, loss function, training loop, and evaluation.
* Data Preprocessing: It includes data transformations like normalization, which is essential for improving model performance.
* Data Loaders: It uses DataLoader for efficient batching and shuffling of data during training.
* Model Definition: It defines a convolutional neural network (MamaNet1) with appropriate layers for image classification.
* Loss Function: It uses nn.CrossEntropyLoss, which is suitable for multi-class classification problems.
* Training Loop: It implements a training loop with epoch-based training and batch-wise updates.
* Evaluation: It includes a validation loop to track the model's performance on unseen data.
* TensorBoard Logging: It uses SummaryWriter to log training metrics to TensorBoard for visualization.
* Model Saving: It saves the model's state dictionary to a file for later use.
Areas for Improvement:
* Optimizer: The script uses optim.Adam as the optimizer. While Adam is a popular choice, you could experiment with other optimizers like SGD with momentum or RMSprop to see if they improve performance.
* Learning Rate: The script doesn't explicitly set a learning rate.
* Regularization: The script doesn't include any regularization techniques like dropout or weight decay. Adding these can help prevent overfitting.
* Early Stopping: The script doesn't implement early stopping, which can help prevent overfitting by stopping training when the validation loss starts to increase.
* Hyperparameter Tuning: The script uses default values for hyperparameters like batch size, epochs, and layer sizes. TODO: perform hyperparameter tuning to find the best values for the model.
* Visualization: The script only logs loss to TensorBoard. You could also log other metrics like accuracy, confusion matrix, and learning rate for a more comprehensive analysis.
* Comments: While the script has some comments, adding more detailed comments would make it easier to understand and maintain.

Optional TODO:

1. Experiment with different model architectures: Try different convolutional layer configurations, pooling strategies, and fully connected layers to see what works best for your data.
2. Use data augmentation: Techniques like random cropping, flipping, and rotation can help improve model generalization.
3. Consider using a pre-trained model: If you have limited data, you can use a pre-trained model on a similar dataset and fine-tune it for your specific task.
'''